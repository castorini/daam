from collections import defaultdict
from pathlib import Path
from typing import List, Type, Any, Dict, Tuple, Set, Union
import math

from diffusers import StableDiffusionPipeline
from diffusers.models.attention import CrossAttention
import numpy as np
import PIL.Image as Image
import torch
import torch.nn.functional as F

from .experiment import GenerationExperiment
from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator
from .utils import compute_token_merge_indices


__all__ = ['trace', 'DiffusionHeatMapHooker', 'HeatMap', 'RawHeatMapCollection']


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
        with torch.cuda.amp.autocast(dtype=torch.float32):
            self.ids_to_heatmaps[(factor, layer_idx, head_idx)] = self.ids_to_heatmaps[(factor, layer_idx, head_idx)] + heatmap

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


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(self, pipeline: StableDiffusionPipeline, low_memory: bool = False):
        self.all_heat_maps = RawHeatMapCollection()
        h = (pipeline.unet.config.sample_size * pipeline.vae_scale_factor)
        self.latent_hw = 4096 if h == 512 else 9216  # 64x64 or 96x96 depending on if it's 2.0-v or 2.0
        self.locator = UNetCrossAttentionLocator(restrict={0} if low_memory else None)
        self.last_prompt: str = ''
        self.last_image: Image = None

        modules = [
            UNetCrossAttentionHooker(
                x,
                self.all_heat_maps,
                layer_idx=idx,
                latent_hw=self.latent_hw
            ) for idx, x in enumerate(self.locator.locate(pipeline.unet))
        ]

        modules.append(PipelineHooker(pipeline, self))

        super().__init__(modules)
        self.pipe = pipeline

    @property
    def layer_names(self):
        return self.locator.layer_names

    def to_experiment(self, path, seed=None, id='.', subtype='.', **compute_kwargs):
        # type: (Union[Path, str], int, str, str, Dict[str, Any]) -> GenerationExperiment
        """Exports the last generation call to a serializable generation experiment."""

        return GenerationExperiment(
            self.last_image,
            self.compute_global_heat_map(**compute_kwargs).heat_maps,
            self.last_prompt,
            seed=seed,
            id=id,
            subtype=subtype,
            path=path,
            tokenizer=self.pipe.tokenizer,
        )

    def compute_global_heat_map(self, prompt=None, factors=None, head_idx=None, layer_idx=None, normalize=False):
        # type: (str, List[float], int, int, bool) -> HeatMap
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            factors: Restrict the application to heat maps with spatial factors in this set. If `None`, use all sizes.
            head_idx: Restrict the application to heat maps with this head index. If `None`, use all heads.
            layer_idx: Restrict the application to heat maps with this layer index. If `None`, use all layers.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.all_heat_maps

        if prompt is None:
            prompt = self.last_prompt

        if factors is None:
            factors = {0, 1, 2, 4, 8, 16, 32, 64}
        else:
            factors = set(factors)

        all_merges = []
        x = int(np.sqrt(self.latent_hw))

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for (factor, layer, head), heat_map in heat_maps:
                if factor in factors and (head_idx is None or head_idx == head) and (layer_idx is None or layer_idx == layer):
                    heat_map = heat_map.unsqueeze(1)
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(x, x), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0)
            except RuntimeError:
                if head_idx is not None or layer_idx is not None:
                    raise RuntimeError('No heat maps found for the given parameters.')
                else:
                    raise RuntimeError('No heat maps found. Did you forget to call `with trace(...)` during generation?')

            maps = maps.mean(0)[:, 0]
            maps = maps[:len(self.pipe.tokenizer.tokenize(prompt)) + 2]  # 1 for SOS and 1 for padding

            if normalize:
                maps = maps / (maps[1:-1].sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return HeatMap(self.pipe.tokenizer, prompt, maps)


class PipelineHooker(ObjectHooker[StableDiffusionPipeline]):
    def __init__(self, pipeline: StableDiffusionPipeline, parent_trace: 'trace'):
        super().__init__(pipeline)
        self.heat_maps = parent_trace.all_heat_maps
        self.parent_trace = parent_trace

    def _hooked_run_safety_checker(hk_self, self: StableDiffusionPipeline, image, *args, **kwargs):
        image, has_nsfw = hk_self.monkey_super('run_safety_checker', image, *args, **kwargs)
        pil_image = self.numpy_to_pil(image)
        hk_self.parent_trace.last_image = pil_image[0]

        return image, has_nsfw

    def _hooked_encode_prompt(hk_self, _: StableDiffusionPipeline, prompt: Union[str, List[str]], *args, **kwargs):
        if not isinstance(prompt, str) and len(prompt) > 1:
            raise ValueError('Only single prompt generation is supported for heat map computation.')
        elif not isinstance(prompt, str):
            last_prompt = prompt[0]
        else:
            last_prompt = prompt

        hk_self.heat_maps.clear()
        hk_self.parent_trace.last_prompt = last_prompt
        ret = hk_self.monkey_super('_encode_prompt', prompt, *args, **kwargs)

        return ret

    def _hook_impl(self):
        self.monkey_patch('run_safety_checker', self._hooked_run_safety_checker)
        self.monkey_patch('_encode_prompt', self._hooked_encode_prompt)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(
            self,
            module: CrossAttention,
            heat_maps: 'RawHeatMapCollection',
            context_size: int = 77,
            layer_idx: int = 0,
            latent_hw: int = 9216
    ):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size
        self.layer_idx = layer_idx
        self.latent_hw = latent_hw

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
        maps = []
        x = x.permute(2, 0, 1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.view(map_.size(0), h, w)
                map_ = map_[map_.size(0) // 2:]  # Filter out unconditional
                maps.append(map_)

        maps = torch.stack(maps, 0)  # shape: (tokens, heads, height, width)
        return maps.permute(1, 0, 2, 3).contiguous()  # shape: (heads, tokens, height, width)

    def _hooked_sliced_attention(hk_self, self, query, key, value, sequence_length, dim):
        batch_size_attention = query.shape[0]
        hidden_states = torch.zeros(
            (batch_size_attention, sequence_length, dim // self.heads), device=query.device, dtype=query.dtype
        )
        slice_size = self._slice_size if self._slice_size is not None else hidden_states.shape[0]
        for i in range(hidden_states.shape[0] // slice_size):
            start_idx = i * slice_size
            end_idx = (i + 1) * slice_size
            attn_slice = torch.baddbmm(
                torch.empty(slice_size, query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query[start_idx:end_idx],
                key[start_idx:end_idx].transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            attn_slice = attn_slice.softmax(dim=-1)
            factor = int(math.sqrt(hk_self.latent_hw // attn_slice.shape[1]))

            if attn_slice.shape[-1] == hk_self.context_size:
                # shape: (batch_size, 64 // factor, 64 // factor, 77)
                maps = hk_self._unravel_attn(attn_slice)

                for head_idx, heatmap in enumerate(maps):
                    hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)

            attn_slice = torch.bmm(attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _hooked_attention(hk_self, self, query, key, value):
        """
        Monkey-patched version of :py:func:`.CrossAttention._attention` to capture attentions and aggregate them.

        Args:
            hk_self (`UNetCrossAttentionHooker`): pointer to the hook itself.
            self (`CrossAttention`): pointer to the module.
            query (`torch.Tensor`): the query tensor.
            key (`torch.Tensor`): the key tensor.
            value (`torch.Tensor`): the value tensor.
        """

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        attn_slice = attention_scores.softmax(dim=-1)
        factor = int(math.sqrt(hk_self.latent_hw // attn_slice.shape[1]))

        if attn_slice.shape[-1] == hk_self.context_size:
            # shape: (batch_size, 64 // factor, 64 // factor, 77)
            maps = hk_self._unravel_attn(attn_slice)

            for head_idx, heatmap in enumerate(maps):
                hk_self.heat_maps.update(factor, hk_self.layer_idx, head_idx, heatmap)

        # compute attention output
        hidden_states = torch.bmm(attn_slice, value)

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _hook_impl(self):
        self.monkey_patch('_attention', self._hooked_attention)
        self.monkey_patch('_sliced_attention', self._hooked_sliced_attention)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
