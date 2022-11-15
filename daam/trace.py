from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import List, Type, Any, Literal, Dict
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


__all__ = ['trace', 'DiffusionHeatMapHooker', 'HeatMap', 'MmDetectHeatMap']


class UNetForwardHooker(ObjectHooker[UNet2DConditionModel]):
    def __init__(self, module: UNet2DConditionModel, heat_maps: defaultdict):
        super().__init__(module)
        self.all_heat_maps = []
        self.heat_maps = heat_maps

    def _hook_impl(self):
        self.monkey_patch('forward', self._forward)

    def _unhook_impl(self):
        pass

    def _forward(hk_self, self, *args, **kwargs):
        super_return = hk_self.monkey_super('forward', *args, **kwargs)
        hk_self.all_heat_maps.append(deepcopy(hk_self.heat_maps))
        hk_self.heat_maps.clear()

        return super_return


class HeatMap:
    def __init__(self, tokenizer: Any, prompt: str, heat_maps: torch.Tensor):
        self.tokenizer = tokenizer
        self.heat_maps = heat_maps
        self.prompt = prompt

    def compute_word_heat_map(self, word: str, word_idx: int = None) -> torch.Tensor:
        merge_idxs = compute_token_merge_indices(self.tokenizer, self.prompt, word, word_idx)
        return self.heat_maps[merge_idxs].mean(0)


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
    def __init__(self, pipeline: StableDiffusionPipeline, weighted: bool = False):
        heat_maps = defaultdict(list)
        modules = [UNetCrossAttentionHooker(x, heat_maps, weighted=weighted) for x in UNetCrossAttentionLocator().locate(pipeline.unet)]
        self.forward_hook = UNetForwardHooker(pipeline.unet, heat_maps)
        modules.append(self.forward_hook)
        super().__init__(modules)

        self.pipe = pipeline

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps

    def compute_global_heat_map(self, prompt, time_weights=None, time_idx=None, last_n=None, factors=None):
        # type: (str, List[float], int, int, List[float]) -> HeatMap
        if time_weights is None:
            time_weights = [1.0] * len(self.forward_hook.all_heat_maps)

        time_weights = np.array(time_weights)
        time_weights /= time_weights.sum()
        all_heat_maps = self.forward_hook.all_heat_maps

        if time_idx is not None:
            heat_maps = [all_heat_maps[time_idx]]
        else:
            heat_maps = all_heat_maps[-last_n:] if last_n is not None else all_heat_maps

        if factors is None:
            factors = {1, 2, 4, 8, 16, 32}
        else:
            factors = set(factors)

        all_merges = []

        for heat_map_map in heat_maps:
            merge_list = []

            for k, v in heat_map_map.items():
                if k in factors:
                    merge_list.append(torch.stack(v, 0).mean(0))

            all_merges.append(merge_list)

        maps = torch.stack([torch.stack(x, 0) for x in all_merges], dim=0)
        maps = maps.sum(0).cuda().sum(2).sum(0)

        return HeatMap(self.pipe.tokenizer, prompt, maps)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(self, module: CrossAttention, heat_maps: defaultdict, context_size: int = 77, weighted: bool = False):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.weighted = weighted

    @torch.no_grad()
    def _up_sample_attn(self, x, value, factor, method='bicubic'):
        # type: (torch.Tensor, torch.Tensor, int, Literal['bicubic', 'conv']) -> torch.Tensor
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)
        value = value.permute(1, 0, 2)
        weights = value.norm(p=1, dim=-1, keepdim=True).unsqueeze(-1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)

                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(64, 64), mode='bicubic')
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1))

        if not self.weighted:
            weights = 1

        maps = torch.stack(maps, 0)

        return (weights * maps).sum(1, keepdim=True).cpu()

    def _hooked_attention(hk_self, self, query, key, value, sequence_length, dim, use_context: bool = True):
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
                if factor >= 1:
                    factor //= 1
                    maps = hk_self._up_sample_attn(attn_slice, value, factor)
                    hk_self.heat_maps[factor].append(maps)
                # print(attn_slice.size(), query.size(), key.size(), value.size())

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
