from collections import defaultdict
from copy import deepcopy
from typing import List, Type, Dict, Any
import math

from diffusers import UNet2DConditionModel, StableDiffusionPipeline
from diffusers.models.attention import CrossAttention
import numpy as np
import torch
import torch.nn.functional as F

from .hook import ObjectHooker, AggregateHooker, UNetCrossAttentionLocator


__all__ = ['trace', 'DiffusionHeatMapHooker']

from .utils import compute_token_merge_indices


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


class DiffusionHeatMapHooker(AggregateHooker):
    def __init__(self, pipeline: StableDiffusionPipeline):
        heat_maps = defaultdict(list)
        modules = [UNetCrossAttentionHooker(x, heat_maps) for x in UNetCrossAttentionLocator().locate(pipeline.unet)]
        self.forward_hook = UNetForwardHooker(pipeline.unet, heat_maps)
        modules.append(self.forward_hook)
        super().__init__(modules)

        self.pipe = pipeline

    @property
    def all_heat_maps(self):
        return self.forward_hook.all_heat_maps

    def compute_word_heat_map(self, word: str, prompt: str, word_idx: int = None, heat_map: torch.Tensor = None, **kwargs) -> torch.Tensor:
        if heat_map is None:
            heat_map = self.compute_global_heat_map(**kwargs)

        merge_idxs = compute_token_merge_indices(self.pipe.tokenizer, prompt, word, word_idx)

        return heat_map[merge_idxs].mean(0)

    def compute_global_heat_map(self, time_weights=None, time_idx=None, last_n=None, factors=None):
        # type: (List[float], int, int, List[float]) -> torch.Tensor
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
        return maps.sum(0).cuda().sum(2).sum(0)


class UNetCrossAttentionHooker(ObjectHooker[CrossAttention]):
    def __init__(self, module: CrossAttention, heat_maps: defaultdict, context_size: int = 77):
        super().__init__(module)
        self.heat_maps = heat_maps
        self.context_size = context_size

    @torch.no_grad()
    def _up_sample_attn(self, x, factor, method: str = 'bicubic'):
        weight = torch.full((factor, factor), 1 / factor ** 2, device=x.device)
        weight = weight.view(1, 1, factor, factor)

        h = w = int(math.sqrt(x.size(1)))
        maps = []
        x = x.permute(2, 0, 1)

        with torch.cuda.amp.autocast(dtype=torch.float32):
            for map_ in x:
                map_ = map_.unsqueeze(1).view(map_.size(0), 1, h, w)
                if method == 'bicubic':
                    map_ = F.interpolate(map_, size=(64, 64), mode="bicubic", align_corners=False)
                    maps.append(map_.squeeze(1))
                else:
                    maps.append(F.conv_transpose2d(map_, weight, stride=factor).squeeze(1).cpu())

        maps = torch.stack(maps, 0).sum(1, keepdim=True).cpu()
        return maps

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

            if use_context and attn_slice.shape[-1] == hk_self.context_size:
                if factor >= 1:
                    factor //= 1
                    maps = hk_self._up_sample_attn(attn_slice, factor)
                    hk_self.heat_maps[factor].append(maps)
                # print(attn_slice.size(), query.size(), key.size(), value.size())

            attn_slice = torch.einsum("b i j, b j d -> b i d", attn_slice, value[start_idx:end_idx])

            hidden_states[start_idx:end_idx] = attn_slice

        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def _hook_impl(self):
        self.monkey_patch('_attention', self._hooked_attention)

    @property
    def num_heat_maps(self):
        return len(next(iter(self.heat_maps.values())))


trace: Type[DiffusionHeatMapHooker] = DiffusionHeatMapHooker
