from typing import List, Generic, TypeVar, Callable, Union, Any
import itertools

from diffusers import UNet2DConditionModel
from diffusers.models.attention import CrossAttention
import torch.nn as nn


ModuleType = TypeVar('ModuleType', bound=nn.Module)
ModuleOutputType = TypeVar('ModuleOutputType')
ModuleInputType = TypeVar('ModuleInputType')


class ModuleLocator(Generic[ModuleType, ModuleInputType, ModuleOutputType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError

    def register_forward_hook(self, model, callback):
        # type: (nn.Module, Callable[[ModuleType, ModuleInputType, ModuleOutputType], None]) -> None
        for module in self.locate(model):
            module.register_forward_hook(callback)


class UnetCrossAttentionLocator(ModuleLocator[CrossAttention, Any, Any]):
    def locate(self, model: UNet2DConditionModel) -> List[CrossAttention]:
        attentions = []

        for block in itertools.chain(model.up_blocks, model.down_blocks, [model.mid_block]):
            attentions.extend(block.attentions)

        return attentions
