from ..loader import *
from ..torch_loader import *
from fastcore.basics import ifnone
from functools import partial


class IOHook:
    def __init__(self, module, depth=None, kwarg_hooks=None):
        self.module = module
        self.hook_handles = []
        self.depth = ifnone(depth, 1)
        self.kwarg_hooks = ifnone(kwarg_hooks, {})
        self.register_hooks()

    def hook_fn(self, module, input, kwargs, output, name):
        Info(f"{name} ({module.__class__.__name__})")
        log_depth = 2
        if hasattr(module, "weight"):
            Info(f"Weights: {module.weight}", depth=log_depth)
        if hasattr(module, "bias"):
            Info(f"Biases: {module.bias}", depth=log_depth)
        if input:
            Info(f"Input: {input}", depth=log_depth)
        if kwargs:
            Info(f"KWargs: {kwargs}", depth=log_depth)
        Info(f"Output: {output}", depth=log_depth)
        line()

    def register_hooks_recursive(self, submodule, prefix, current_depth=1):
        if self.depth is None or current_depth <= self.depth:
            hook_handle = submodule.register_forward_hook(
                partial(self.hook_fn, name=prefix),
                with_kwargs=True,
            )
            Info(f"Registered {submodule.__class__.__name__} @ {prefix}")
            self.hook_handles.append(hook_handle)

        if self.depth is None or current_depth < self.depth:
            for name, subsubmodule in submodule.named_children():
                self.register_hooks_recursive(
                    subsubmodule,
                    prefix=prefix + "." + name,
                    current_depth=current_depth + 1,
                )

    def register_hooks(self):
        self.register_hooks_recursive(self.module, prefix="")

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
