from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Type, Any, Literal, Dict, Tuple, Set
import math

from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.attention import CrossAttention
import numba
import numpy as np
import torch
import torch.nn.functional as F

from .experiment import COCO80_LABELS
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator
from .utils import compute_token_merge_indices


__all__ = ['trace', 'DiffusionHeatMapHooker', 'HeatMap', 'MmDetectHeatMap', 'RawHeatMapCollection']


class HeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt

    def compute_word_heat_map(self, word: str, word_idx: int = None) -> torch.Tensor:
        merge_idxs = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx)
        return self.heat_maps[merge_idxs].mean(0)


RawHeatMapKey = Tuple[int, int, int]  # factor, layer, head

class RawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)

    def update(self, factor: int, layer_idx: int, head_idx: int, heatmap: torch.Tensor):
        self.ids_to_heatmaps[(factor, layer_idx, head_idx)] += heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()

class MmDetectHeatMap:
    def __init__(self, pred_file: str | Path, threshold: float = 0.95):
        @numba.njit
        def _compute_mask(masks: np.ndarray, bboxes: np.ndarray):
            x_any = np.any(masks, axis=1)
            y_any = np.any(masks, axis=2)
            num_masks = len(bboxes)

            for idx in range(num_masks):
                x = np.where(x_any[idx, :])[0]
                y = np.where(y_any[idx, :])[0]
                bboxes[idx, :4] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1], dtype=np.float32)

        pred_file = Path(pred_file)
        self.word_masks: Dict[str, torch.Tensor] = defaultdict(lambda: 0)
        bbox_result, masks = torch.load(pred_file)
        labels = [np.full(bbox.shape[0], i, dtype=np.int32) for i, bbox in enumerate(bbox_result)]
        labels = np.concatenate(labels)
        bboxes = np.vstack(bbox_result)

        if masks is not None and bboxes[:, :4].sum() == 0:
            _compute_mask(masks, bboxes)
            scores = bboxes[:, -1]
            inds = scores > threshold
            labels = labels[inds]
            masks = masks[inds, ...]

            for lbl, mask in zip(labels, masks):
                self.word_masks[COCO80_LABELS[lbl]] |= torch.from_numpy(mask)

            self.word_masks = {k: v.float() for k, v in self.word_masks.items()}

    def compute_word_heat_map(self, word: str) -> torch.Tensor:
        return self.word_masks[word]


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(self, pipeline: StableDiffusionPipeline, low_memory: bool = False):
        self.all_heat_maps = RawHeatMapCollection()
        locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None)
        modules = [UNetCrossAttentionHooker(x, self.all_heat_maps, layer_idx=idx) for idx, x in enumerate(locator.locate(pipeline.unet))]
        super().__init__(modules)

        self.pipe = pipeline

    def compute_global_heat_map(self, prompt, factors=None, head_idx=None, layer_idx=None):
        # type: (str, List[float], int, int) -> HeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if factors is None:
            factors = {1, 2, 4, 8, 16, 32}
        else:
            factors = set(factors)

        all_merges = []

        for ((factor, layer, head), heat_map) in heat_maps:
            if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                heat_map = heat_map.unsqueeze(1)
                all_merges.append(F.interpolate(heat_map, size=(64, 64), mode='bicubic'))

        maps = torch.stack(all_merges, dim=0)
        print(maps.shape)
        maps = maps.sum(0)[:, 0]

        return HeatMap(self.pipe.tokenizer, prompt, maps)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(self, module: CrossAttention, heat_maps: 'RawHeatMapCollection', context_size: int = 77, layer_idx: int = 0):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx

    @torch.no_grad()
    def _unravel_attn(self, x):
        # type: (torch.Tensor) -> torch.Tensor
        # x shape: (heads, height * width, tokens)
        """
        Unravels the attention, returning it as a collection of heat maps.

        Args:
            x (`torch.Tensor`): cross attention slice/map between the words and the tokens.
            value (`torch.Tensor`): the value tensor.

        Returns:
            `List[Tuple[int, torch.Tensor]]`: the list of heat maps across heads.
        """
        h = w = int(math.sqrt(x.size(1)))
        x = x.permute(0, 2, 1).contiguous()
        return x.view(x.size(0), -1, h, w)

    def _hooked_attention(hk_self, self, query, key, value, sequence_length, dim, use_context: bool = True):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            hk_self (`UNetCrossAttentionHooker`): pointer to the hook itself.
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
            use_context (`bool`): whether to check if the resulting attention slices are between the words and the image
        """
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = (
                    torch.einsum("b i d, b j d -> b i j", query[start_idx:end_idx], key[start_idx:end_idx]) * self.scale
            )
            factor = int(math.sqrt(4096 // attn_slice.shape[1]))
            attn_slice = attn_slice.softmax(-1)
            hid_states = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

            if use_context and attn_slice.shape[-1] == hk_self.context_size:
                # shape: (batch_size, 64 // factor, 64 // factor, 77)
                maps = hk_self._unravel_attn(attn_slice)

                for head_idx, heatmap in enumerate(maps):
                    hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)

            hidden_states[start_idx:end_idx] = hid_states

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _hook_impl(self):
        self.monkey_patch('_attention', self._hooked_attention)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
