from typing import List

from diffusers.models.attention import CrossAttention


class trace:
    def __init__(self, modules: List[CrossAttention], time_weights: List[float] = None):
        self.modules = modules
        self.time_weights = time_weights
        self.handles = []

    def __enter__(self):
        for mod in self.modules:
            handles.append(mod.register_forward_hook())

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()

        self.handles.clear()
