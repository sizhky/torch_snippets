# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/misc.ipynb.

# %% auto 0
__all__ = ['Timer', 'track2', 'summarize_input', 'timeit', 'io', 'tryy', 'cast_inputs']

# %% ../nbs/misc.ipynb 2
import time
from .logger import Debug, Warn, debug_mode, Info, Trace, trace_mode
from .markup2 import AD
from functools import wraps
from typing import get_type_hints
from fastcore.basics import ifnone
from fastcore.foundation import L

# %% ../nbs/misc.ipynb 3
class Timer:
    def __init__(self, N, smooth=True, mode=1):
        "print elapsed time every iteration and print out remaining time"
        "assumes this timer is called exactly N times or less"
        self.tok = self.start = time.time()
        self.N = N
        assert N is None or N >= 0, "N should be None or a non-negative integer"
        self.ix = 0
        self.smooth = smooth
        self.mode = mode
        # 0 = instant-speed, i.e., time remaining is a funciton of only the last iteration
        # Useful when you know that each loop takes unequal time (increasing/decreasing speeds)
        # 1 = average, i.e., time remaining is a function of average of all iterations
        # Usefule when you know on average each loop or a group of loops take around the same time

    def __call__(self, *, ix=None, info=None):
        ix = self.ix if ix is None else ix
        info = "" if info is None else f"{info}\t"
        tik = time.time()
        elapsed = tik - self.start

        _total = self.N if self.N is not None else "*"

        if self.mode == 0:
            ielapsed = tik - self.tok
            ispeed = ielapsed
            if self.N is not None:
                iremaining = (self.N - (ix + 1)) * ispeed
            else:
                iremaining = -1

            iunit = "s/iter"
            if ispeed < 1:
                ispeed = 1 / ispeed
                iunit = "iters/s"

            iestimate = iremaining + elapsed
            _remaining = f"- {iremaining:.2f}s remaining " if self.N is not None else ""

            _info = f"{info}{ix+1}/{_total} ({elapsed:.2f}s {_remaining}- {ispeed:.2f} {iunit})"

        else:
            speed = elapsed / (ix + 1)
            if self.N is not None:
                remaining = (self.N - (ix + 1)) * speed
            else:
                remaining = -1

            unit = "s/iter"
            if speed < 1:
                speed = 1 / speed
                unit = "iters/s"
            estimate = remaining + elapsed
            _remaining = f" - {remaining:.2f}s remaining " if self.N is not None else ""
            _info = f"{info}{ix+1}/{_total} ({elapsed:.2f}s {_remaining}- {speed:.2f} {unit})"

        print(
            _info + " " * 10,
            end="\r",
        )
        self.ix += 1
        self.tok = tik


def track2(iterable, *, total=None, info_prefix=None):
    info_prefix = ifnone(info_prefix, "")
    try:
        total = ifnone(total, len(iterable))
    except Exception as e:
        Warn(f"Unable to get length of iterable: {e}")
    timer = Timer(total)
    for item in iterable:
        info = yield item
        _info = (
            f"{info_prefix} {info}"
            if info and info_prefix
            else info or info_prefix or None
        )
        timer(info=_info)
        if info is not None:
            yield  # Just to ensure the send operation stops


# %% ../nbs/misc.ipynb 10
def summarize_input(args, kwargs, outputs=None):
    o = AD(args, kwargs)
    if outputs is not None:
        o.outputs = outputs
    return o.summary()


def timeit(func):
    def inner(*args, **kwargs):
        s = time.time()
        o = func(*args, **kwargs)
        Info(f"{time.time() - s:.2f} seconds to execute `{func.__name__}`")
        return o

    return inner


def io(func=None, *, level="debug"):
    logfuncs = {
        "debug": lambda i: Debug(i, depth=2),
        "info": lambda i: Info(i, depth=2),
        "trace": lambda i: Trace(i, depth=2),
    }
    try:
        logfunc = logfuncs[level.lower()]
    except KeyError:
        raise ValueError(f"level should be one of {list(logfuncs.keys())}")

    def decorator(func):
        def inner(*args, **kwargs):
            if level == "trace":
                try:
                    from pysnooper import snoop
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "`pip install pysnooper` if you want to trace the function line by line"
                    )
                _func = snoop()(func)
            else:
                _func = func
            s = time.time()
            o = _func(*args, **kwargs)
            info = f"""
{time.time() - s:.2f} seconds to execute `{func.__name__}`
{summarize_input(args=args, kwargs=kwargs, outputs=o)}
            """
            logfunc(info)
            return o

        return inner

    if func is None:
        o = decorator
    else:
        o = decorator(func)
    return o

# %% ../nbs/misc.ipynb 18
def tryy(
    func=None,
    *,
    output_to_return_on_fail=None,
    silence_errors=False,
    print_traceback=False,
    store_errors: bool = True,
):
    def decorator(f):
        if isinstance(store_errors, bool) and store_errors:
            error_store = []
        elif isinstance(store_errors, (list, type([]))):  # Avoid `L` unless defined
            error_store = store_errors

        @wraps(f)  # Preserve the original function's metadata
        def wrapper(*args, **kwargs):
            try:
                return f(*args, **kwargs)
            except Exception as e:
                if not silence_errors:
                    if not print_traceback:
                        tb = f"{type(e).__name__}: {str(e)}"
                        Warn(f"Error for `{f.__name__}`: {tb}")
                    else:
                        import traceback

                        tb = traceback.format_exc()
                        Warn(f"Error for `{f.__name__}`:\n{tb}")
                else:
                    tb = None
                if store_errors is not None:
                    error_store.append(
                        {
                            "func": f.__name__,
                            "args": args,
                            "kwargs": kwargs,
                            "tb": tb,
                            "err_type": type(e).__name__,
                        }
                    )
                if callable(output_to_return_on_fail):
                    return output_to_return_on_fail(*args, **kwargs)
                return output_to_return_on_fail

        # Attach additional attributes
        wrapper.F = f
        wrapper.error_store = error_store

        def errors():
            import pandas as pd

            return pd.DataFrame(error_store)

        wrapper.errors = errors
        wrapper.error_summary = errors  # backward compatibility
        return wrapper

    if callable(func):
        return decorator(func)
    return decorator

# %% ../nbs/misc.ipynb 40
def cast_inputs(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get type hints of the function
        type_hints = get_type_hints(func)

        # Get the function arguments as a dictionary
        from inspect import signature

        bound_args = signature(func).bind(*args, **kwargs)
        bound_args.apply_defaults()

        # Cast arguments to the specified type if possible
        for arg_name, arg_value in bound_args.arguments.items():
            if arg_name in type_hints:
                target_type = type_hints[arg_name]
                try:
                    bound_args.arguments[arg_name] = target_type(arg_value)
                except (ValueError, TypeError):
                    raise TypeError(f"Cannot cast `{arg_name}` ({arg_value}) to {target_type}")

        # Call the function with the cast arguments
        return func(*bound_args.args, **bound_args.kwargs)

    return wrapper

