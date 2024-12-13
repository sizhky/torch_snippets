import traceback
from contextlib import contextmanager
from functools import partial

from fastcore.basics import ifnone

from ..loader import *
from ..markup2 import AD
from ..torch_loader import *


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


# skip = {'SiLU', 'LayerNorm'}
# keep = {'CrossAttnDownBlock2D', 'CrossAttnUpBlock2D', 'UNetMidBlock2DCrossAttn', 'UpBlock2D', 'DownBlock2D'}
def print_shapes_hook(
    module, input, kwargs, output, skip=None, keep=None, child_name=""
):
    from builtins import print

    skip = set() if skip is None else set(skip)  # Modules to exclude while printing
    keep = set() if keep is None else set(keep)  # Modules to include while printing
    try:
        name = module.__class__.__name__
        if name in skip:
            return
        if keep != {"all"} and name not in keep:
            return
        line()
        print(f"Module Name: {name}. Child Name: {child_name}")
        print(f"Input Kwargs: ")
        if "keys" in kwargs:  # kwargs literlly has the key called `key`
            kwargs["keys_"] = kwargs["keys"]
            del kwargs["keys"]
        print(AD(kwargs).summary(depth=1))
        print(f"Input Args:")
        if isinstance(input, (list, tuple, set)):
            input = {str(i + 1): v for i, v in enumerate(input)}
            print(AD(input).summary(depth=1))
        else:
            input = {"1": input}
            print(AD(input).summary(depth=1))
        print("Outputs: ")
        if isinstance(output, (list, tuple, set)):
            output = {str(i + 1): v for i, v in enumerate(output)}
            print(AD(output).summary(depth=1))
        else:
            output = {"1": output}
            print(AD(output).summary(depth=1))
        line()
    except Exception as e:
        print(f"ERROR: {e} @ {name}")


def attach_hooks(
    model,
    hook=print_shapes_hook,
    required_children=None,
    print_only=None,
    all_children=False,
    parent_name="",
):
    """Function to attach the hooks and return handles"""
    # handles = [model.register_forward_hook(hook, with_kwargs=True)]
    handles = []
    for child_name, layer in model.named_children():
        _child_name = f"{parent_name}.{child_name}" if parent_name != "" else child_name
        if (
            all_children
            or (required_children is not None and _child_name in required_children)
            or (layer.__class__.__name__ in print_only)
        ):
            _hook = partial(
                hook,
                keep={layer.__class__.__name__},
                child_name=_child_name,
            )
            handle = layer.register_forward_hook(_hook, with_kwargs=True)
            handles.append(handle)
        if len(list(layer.children())) > 0:
            handles.extend(
                attach_hooks(
                    layer,
                    hook=hook,
                    required_children=required_children,
                    print_only=print_only,
                    parent_name=_child_name,
                    all_children=all_children,
                )
            )
    return handles


def detach_hooks(handles):
    """Function to detach hooks"""
    for handle in handles:
        handle.remove()


@contextmanager
def hook_context_for(module, hook_fn, **kwargs):
    try:
        hook_fn = partial(hook_fn, **kwargs)
        handles = attach_hooks(module, hook_fn)
        yield module
        detach_hooks(handles)
    except Exception as e:
        Warn(f"Error: {e}")
    finally:
        try:
            detach_hooks(handles)
        except:
            # nothing needs to be done
            ...


@contextmanager
def print_module_ios_for(module, print_only={}, required_children=None):
    if print_only is None:
        print_only = {module.__class__.__name__}
    try:
        # _print_shapes_hook = partial(print_shapes_hook, keep=print_only)
        handles = attach_hooks(
            module,
            print_shapes_hook,
            required_children=required_children,
            print_only=print_only,
        )
        yield module
        detach_hooks(handles)
    except Exception as e:
        Warn(f"Error: {e}\n{traceback.format_exc()}")
    finally:
        try:
            detach_hooks(handles)
        except:
            # nothing needs to be done
            ...
