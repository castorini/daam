from typing import List, Generic, TypeVar, Callable, Union, Any
import functools
import itertools

from diffusers import UNet2DConditionModel
from diffusers.models.attention import CrossAttention
import torch.nn as nn


__all__ = ['ObjectHooker', 'ModuleLocator', 'AggregateHooker', 'UNetCrossAttentionLocator']


ModuleType = TypeVar('ModuleType')
ModuleListType = TypeVar('ModuleListType', bound=List)


class ModuleLocator(Generic[ModuleType]):
    def locate(self, model: nn.Module) -> List[ModuleType]:
        raise NotImplementedError


class ObjectHooker(Generic[ModuleType]):
    def __init__(self, module: ModuleType):
        self.module: ModuleType = module
        self.hooked = False
        self.old_state = dict()

    def __enter__(self):
        self.hook()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.unhook()

    def hook(self):
        if self.hooked:
            raise RuntimeError('Already hooked module')

        self.old_state = dict()
        self.hooked = True
        self._hook_impl()

        return self

    def unhook(self):
        if not self.hooked:
            raise RuntimeError('Module is not hooked')

        for k, v in self.old_state.items():
            if k.startswith('old_fn_'):
                setattr(self.module, k[7:], v)

        self.hooked = False
        self._unhook_impl()

        return self

    def monkey_patch(self, fn_name, fn):
        self.old_state[f'old_fn_{fn_name}'] = getattr(self.module, fn_name)
        setattr(self.module, fn_name, functools.partial(fn, self.module))

    def monkey_super(self, fn_name, *args, **kwargs):
        return self.old_state[f'old_fn_{fn_name}'](*args, **kwargs)

    def _hook_impl(self):
        raise NotImplementedError

    def _unhook_impl(self):
        pass


class AggregateHooker(ObjectHooker[ModuleListType]):
    def _hook_impl(self):
        for h in self.module:
            h.hook()

    def _unhook_impl(self):
        for h in self.module:
            h.unhook()

    def register_hook(self, hook: ObjectHooker):
        self.module.append(hook)


class UNetCrossAttentionLocator(ModuleLocator[CrossAttention]):
    def __init__(self, restrict: bool = None):
        self.restrict = restrict
        self.layer_names = []

    def locate(self, model: UNet2DConditionModel) -> List[CrossAttention]:
        """
        Locate all cross-attention modules in a UNet2DConditionModel.

        Args:
            model (`UNet2DConditionModel`): The model to locate the cross-attention modules in.

        Returns:
            `List[CrossAttention]`: The list of cross-attention modules.
        """
        self.layer_names.clear()
        blocks_list = []
        up_names = ['up'] * len(model.up_blocks)
        down_names = ['down'] * len(model.up_blocks)

        # The middle block was not found to be useful.
        for unet_block, name in itertools.chain(
                zip(model.up_blocks, up_names),
                zip(model.down_blocks, down_names)
        ):
            if 'CrossAttn' in unet_block.__class__.__name__:
                blocks = []

                for spatial_transformer in unet_block.attentions:
                    for transformer_block in spatial_transformer.transformer_blocks:
                        blocks.append(transformer_block.attn2)

                blocks = [b for idx, b in enumerate(blocks) if self.restrict is None or idx in self.restrict]
                names = [f'{name}-attn-{i}' for i in range(len(blocks)) if self.restrict is None or i in self.restrict]
                blocks_list.extend(blocks)
                self.layer_names.extend(names)

        return blocks_list
