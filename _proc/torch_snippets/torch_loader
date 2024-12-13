"v0.15.41"
"""Todo - v0.15.5
- [ ] accumulate lists of tensors
- [ ] make plot_epochs even faster
"""

__all__ = [
    "torch",
    "th",
    "torchvision",
    "T",
    "transforms",
    "nn",
    "np",
    "F",
    "Dataset",
    "DataLoader",
    "optim",
    "Report",
    "Reshape",
    "Permute",
    "device",
    "save_torch_model_weights_from",
    "load_torch_model_weights_to",
    "detach",
    "cat_with_padding",
]

import re
import time
from collections import defaultdict, namedtuple
from itertools import dropwhile, takewhile
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from loguru import logger
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import makedir, os

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import wandb
except ImportError:
    pass

try:
    from mlflow_extend import mlflow
except ImportError:
    mlflow = None


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Permute(nn.Module):
    def __init__(self, *order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)


metric = namedtuple("metric", "pos,val".split(","))


def process(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
    return v


def info(report, precision):
    report = {k: process(v) for k, v in report.items()}
    return "  ".join([f"{k}: {v:.{precision}f}" for k, v in report.items()])


def to_np(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().numpy())
    else:
        return x


class Report:
    def __init__(
        self,
        n_epochs=None,
        precision=3,
        old_report=None,
        wandb_project=None,
        hyper_parameters=None,
        **kwargs,
    ):
        self.start = time.time()
        self.n_epochs = n_epochs
        self.precision = precision
        self.completed_epochs = -1
        self.logged = set()
        self.set_external_logging(wandb_project, hyper_parameters)
        if old_report:
            self.prepend(old_report)

    def reset_time(self):
        self.start = time.time()

    def prepend(self, old_report):
        for k in old_report.logged:
            self.logged.append(k)
            last = getattr(old_report, k)[-1].pos
            setattr(self, k, [])
            for m in getattr(old_report, k):
                getattr(self, k).append(metric(m.pos - last, to_np(m.val)))

    def record(self, pos, **metrics):
        metrics = {k: to_np(v) for k, v in metrics.items()}
        for k, v in metrics.items():
            if k in ["end", "pos"]:
                continue
            if hasattr(self, k):
                getattr(self, k).append(metric(pos, to_np(v)))
            else:
                setattr(self, k, [])
                getattr(self, k).append(metric(pos, to_np(v)))
                self.logged.add(k)

        if not any(["val" in key for key in metrics.keys()]):
            key = "train_step"
            step = self.train_step
        else:
            key = "validation_step"
            step = self.validation_step

        self.report_metrics(pos, **metrics)

        metrics = {k: v for k, v in metrics.items() if not isinstance(v, str)}
        if self.wandb_logging:
            wandb.log({**metrics, key: step})
        if self.mlflow_logging:
            mlflow.log_metrics(metrics, step=step)

        if key == "train_step":
            self.train_step += 1
        else:
            self.validation_step += 1

    def plot(self, keys: Union[List, str] = None, smooth=0, ax=None, **kwargs):
        _show = True if ax is None else False
        if ax is None:
            sz = 8, 6
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", sz))

        keys = self.logged if keys is None else keys
        if isinstance(keys, str):
            key_pattern = keys
            keys = [key for key in self.logged if re.search(key_pattern, key)]

        for k in keys:
            xs, ys = list(zip(*getattr(self, k)))
            if smooth:
                ys = moving_average(np.array(ys), smooth)

            if "val" in k:
                _type = "--"
            elif "test" in k:
                _type = ":"
            else:
                _type = "-"

            ax.plot(xs, ys, _type, label=k)
        ax.grid(True)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metrics")
        ax.set_title(
            kwargs.get("title", None), fontdict=kwargs.get("fontdict", {"size": 20})
        )
        if kwargs.get("log", False):
            ax.semilogy()
        ax.legend()
        if _show:
            plt.show()

    def history(self, k):
        return [v for _, v in getattr(self, k)]

    def plot_epochs(self, keys: list = None, ax=None, **kwargs):
        _show = True if ax is None else False
        if ax is None:
            sz = 8, 6
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", sz))
        avgs = defaultdict(list)
        keys = self.logged if keys is None else keys

        from tqdm import trange

        if isinstance(keys, str):
            key_pattern = keys
            keys = [key for key in self.logged if re.search(key_pattern, key)]
        xs = []
        for epoch in trange(-100, self.n_epochs + 1):
            for k in keys:
                items = takewhile(
                    lambda x: epoch - 1 <= x.pos < epoch,
                    dropwhile(
                        lambda x: (epoch - 1 > x.pos or x.pos > epoch), getattr(self, k)
                    ),
                )
                items = list(items)
                if items == []:
                    continue
                xs.append(epoch)
                avgs[k].append(np.mean([v for pos, v in items]))
            xs = sorted(set(xs))

        for k in avgs:
            if "val" in k:
                _type = "--"
            elif "test" in k:
                _type = ":"
            else:
                _type = "-"
            if len(avgs[k]) != len(xs):
                logger.info(
                    f"metric {k} was not fully recorded. Plotting final epochs using last recorded value"
                )
                avgs[k].extend([avgs[k][-1]] * (len(xs) - len(avgs[k])))
            ax.plot(
                xs,
                avgs[k],
                _type,
                label=k,
            )
        ax.grid(True)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metrics")
        ax.set_title(
            kwargs.get("title", None), fontdict=kwargs.get("fontdict", {"size": 20})
        )
        if kwargs.get("log", False):
            ax.semilogy()
        plt.legend()
        if _show:
            plt.show()

    def report_avgs(self, epoch, return_avgs=True, end="\n"):
        avgs = {}
        for k in self.logged:
            avgs[k] = np.mean(
                [v for pos, v in getattr(self, k) if epoch - 1 <= pos < epoch]
            )
        self.report_metrics(epoch, end=end, **avgs)
        avgs = {f"epoch_{k}": v for k, v in avgs.items()}
        if self.wandb_logging:
            wandb.log({**avgs, "epoch": epoch})
        if self.mlflow_logging:
            mlflow.log_metrics(avgs, step=epoch)
        if return_avgs:
            return avgs

    def report_metrics(self, pos, **report):
        """Report training and validation metrics
        Required variables to be initialized before calling this function:
        1. start (time.time())
        2. n_epochs (int)

        Special kwargs:
        1. end - line ending after print (default newline)
        2. log - prefix info before printing summary

        Special argument:
        1. pos - position in training/testing process - float between 0 - n_epochs

        Usage:
        report_metrics(pos=1.42, train_loss=train_loss, validation_loss=validation_loss, ... )
        where each kwarg is a float
        """
        elapsed = time.time() - self.start
        end = report.pop("end", "\n")
        log = report.pop("log", "")
        log = log + ": " if log != "" else log
        elapsed = "  ({:.2f}s - {:.2f}s remaining)".format(
            time.time() - self.start, ((self.n_epochs - pos) / pos) * elapsed
        )
        current_iteration = f"EPOCH: {pos:.3f}  "
        if end == "\r":
            print(
                f"\r{log}{current_iteration}{info(report, self.precision)}{elapsed}",
                end="",
            )
        else:
            print(
                f"\r{log}{current_iteration}{info(report, self.precision)}{elapsed}",
                end=end,
            )

    def set_external_logging(self, project=None, hyper_parameters=None):
        if project is not None:
            if "/" in project:
                project, name = project.split("/")
            self.wandb_logging = True
            wandb.init(project=project, name=name, config=hyper_parameters)
        else:
            self.wandb_logging = False

        if mlflow and mlflow.active_run():
            self.mlflow_logging = True
        else:
            self.mlflow_logging = False

        self.train_step = 0
        self.validation_step = 0

    def finish_run(self, **kwargs):
        if not kwargs.get("do_not_finish_wandb", False):
            if self.wandb_logging:
                wandb.finish()


try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.progress import ProgressBarBase

    class LightningReport(ProgressBarBase):
        def __init__(
            self,
            epochs,
            print_every=None,
            print_total=None,
            precision=4,
            old_report=None,
        ):
            super().__init__()
            self.enable = True
            self.epoch_ix = 0
            _report = old_report.report if old_report is not None else None
            self.report = Report(epochs, precision, _report)
            if print_every is not None:
                self.print_every = print_every
            else:
                if print_total is None:
                    if epochs < 11:
                        self.print_every = 1
                else:
                    self.print_every = epochs // print_total

        def disable(self):
            self.enable = False

        def on_epoch_end(self, trainer, pl_module):
            self.epoch_ix += 1
            if self.epoch_ix % self.print_every == 0:
                self.report.report_avgs(self.epoch_ix)

        def prefix_dict_keys(self, prefix, _dict):
            _d = {}
            for k in _dict:
                _d[f"{prefix}_{k}"] = _dict[k]
            return _d

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        ):
            super().on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
            if isinstance(outputs, list):
                loss = float(trainer.progress_bar_dict["loss"])
                outputs = outputs[0][0]["extra"]
                outputs["loss"] = loss
            outputs = self.prefix_dict_keys("trn", outputs)
            self.px = self.epoch_ix + ((1 + batch_idx) / self.total_train_batches)
            self.report.record(self.px, end="\r", **outputs)

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        ):
            super().on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
            loss = float(trainer.progress_bar_dict["loss"])
            self.px = self.epoch_ix + ((1 + batch_idx) / self.total_val_batches)
            outputs = self.prefix_dict_keys("val", outputs)
            self.report.record(self.px, end="\r", **outputs)

        def __getattr__(self, attr, **kwargs):
            return getattr(self.report, attr, **kwargs)

    __all__ += ["LightningReport", "pl"]
except Exception as e:
    ...


def moving_average(a, n=3):
    b = np.zeros_like(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    _n = len(b) - n
    b[-_n - 1 :] = ret[(n - 1) :] / n
    return b


def save_torch_model_weights_from(model, fpath):
    "from model to fpath"
    torch.save(model.state_dict(), fpath)
    fsize = os.path.getsize(fpath) >> 20
    logger.opt(depth=1).log("INFO", f"Saved weights of size ~{fsize} MB to {fpath}")


def load_torch_model_weights_to(model, fpath, device=None):
    "to model from fpath"
    if not device:
        model.load_state_dict(torch.load(fpath))
    else:
        model.load_state_dict(torch.load(fpath, map_location=device))
    logger.opt(depth=1).log("INFO", f"Loaded weights from {fpath} to given model")


def detach(i):
    if isinstance(i, dict):
        for k, v in i.items():
            i[k] = detach(v)
        return i
    elif isinstance(i, (list, tuple)):
        return [detach(j) for j in i]
    else:
        return i.cpu().detach()


def cat_with_padding(tensors, mode="constant", value=-100):
    """
    Concatenates a list of tensors with padding to make them compatible along 0th dimension.

    Args:
        tensors (list of torch.Tensor): List of tensors to be concatenated.
        mode (str, optional): The padding mode. Default is "constant".
        value (int, optional): The padding value for "constant" mode. Default is -100.

    Returns:
        torch.Tensor: Concatenated tensor with padding to match dimensions.

    Note:
        - All tensors should have the same dimension except for the stacking dimension.
        - Padding is added to match the dimensions of the largest tensor in the specified stacking dimension.
    """
    assert all(
        [tensors[0].ndim == t.ndim for t in tensors[1:]]
    ), "All tensors should have the same number of dimensions"
    sizes = [(t.size()[1:]) for t in tensors]
    full_size = [max(s) for s in list(zip(*sizes))]
    if full_size == []:
        # Given tensors are just scalars
        return torch.stack(tensors)

    def make_padding(current_size, target_size):
        o = ()
        for curr_d, targ_d in zip(current_size[::-1], target_size[::-1]):
            diff_d = targ_d - curr_d
            o = o + (0, diff_d)
        return o

    out = torch.cat(
        [
            F.pad(x, make_padding(x.size()[1:], full_size), mode=mode, value=value)
            for x in tensors
        ],
        dim=0,
    )
    return out


"v0.15.41"
"""Todo - v0.15.5
- [ ] accumulate lists of tensors
- [ ] make plot_epochs even faster
"""

__all__ = [
    "torch",
    "th",
    "torchvision",
    "T",
    "transforms",
    "nn",
    "np",
    "F",
    "Dataset",
    "DataLoader",
    "optim",
    "Report",
    "Reshape",
    "Permute",
    "device",
    "save_torch_model_weights_from",
    "load_torch_model_weights_to",
    "detach",
    "cat_with_padding",
    "clean_gpu_mem",
    "get_latest_checkpoint",
]

import gc
import re
import time
import traceback
from collections import defaultdict, namedtuple
from itertools import dropwhile, takewhile
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch as th
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from loguru import logger
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .paths import makedir, os

device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    import wandb
except ImportError:
    pass

try:
    from mlflow_extend import mlflow
except ImportError:
    mlflow = None


class Reshape(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(*self.shape)


class Permute(nn.Module):
    def __init__(self, *order):
        super().__init__()
        self.order = order

    def forward(self, x):
        return x.permute(*self.order)


metric = namedtuple("metric", "pos,val".split(","))


def process(v):
    if isinstance(v, torch.Tensor):
        if v.numel() == 1:
            return v.item()
    return v


def info(report, precision):
    report = {k: process(v) for k, v in report.items()}
    return "  ".join([f"{k}: {v:.{precision}f}" for k, v in report.items()])


def to_np(x):
    if isinstance(x, torch.Tensor):
        return float(x.detach().cpu().numpy())
    else:
        return x


class Report:
    def __init__(
        self,
        n_epochs=None,
        precision=3,
        old_report=None,
        wandb_project=None,
        hyper_parameters=None,
        **kwargs,
    ):
        self.start = time.time()
        self.n_epochs = n_epochs
        self.precision = precision
        self.completed_epochs = -1
        self.logged = set()
        self.set_external_logging(wandb_project, hyper_parameters)
        if old_report:
            self.prepend(old_report)

    def reset_time(self):
        self.start = time.time()

    def prepend(self, old_report):
        for k in old_report.logged:
            self.logged.add(k)
            last = getattr(old_report, k)[-1].pos
            setattr(self, k, [])
            for m in getattr(old_report, k):
                getattr(self, k).append(metric(m.pos - last, to_np(m.val)))

    def record(self, pos, **metrics):
        metrics = {k: to_np(v) for k, v in metrics.items()}
        for k, v in metrics.items():
            if k in ["end", "pos"]:
                continue
            if hasattr(self, k):
                getattr(self, k).append(metric(pos, to_np(v)))
            else:
                setattr(self, k, [])
                getattr(self, k).append(metric(pos, to_np(v)))
                self.logged.add(k)

        if not any(["val" in key for key in metrics.keys()]):
            key = "train_step"
            step = self.train_step
        else:
            key = "validation_step"
            step = self.validation_step

        self.report_metrics(pos, **metrics)

        metrics = {k: v for k, v in metrics.items() if not isinstance(v, str)}
        if self.wandb_logging:
            wandb.log({**metrics, key: step})
        if self.mlflow_logging:
            mlflow.log_metrics(metrics, step=step)

        if key == "train_step":
            self.train_step += 1
        else:
            self.validation_step += 1

    def plot(self, keys: Union[List, str] = None, smooth=0, ax=None, **kwargs):
        _show = True if ax is None else False
        if ax is None:
            sz = 8, 6
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", sz))

        keys = self.logged if keys is None else keys
        if isinstance(keys, str):
            key_pattern = keys
            keys = [key for key in self.logged if re.search(key_pattern, key)]

        for k in keys:
            xs, ys = list(zip(*getattr(self, k)))
            if smooth:
                ys = moving_average(np.array(ys), smooth)

            if "val" in k:
                _type = "--"
            elif "test" in k:
                _type = ":"
            else:
                _type = "-"

            ax.plot(xs, ys, _type, label=k)
        ax.grid(True)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metrics")
        ax.set_title(
            kwargs.get("title", None), fontdict=kwargs.get("fontdict", {"size": 20})
        )
        if kwargs.get("log", False):
            ax.semilogy()
        ax.legend()
        if _show:
            plt.show()

    def history(self, k):
        return [v for _, v in getattr(self, k)]

    def plot_epochs(self, keys: list = None, ax=None, **kwargs):
        _show = True if ax is None else False
        if ax is None:
            sz = 8, 6
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", sz))
        avgs = defaultdict(list)
        keys = self.logged if keys is None else keys

        from tqdm import trange

        if isinstance(keys, str):
            key_pattern = keys
            keys = [key for key in self.logged if re.search(key_pattern, key)]
        xs = []
        for epoch in trange(-100, self.n_epochs + 1):
            for k in keys:
                items = takewhile(
                    lambda x: epoch - 1 <= x.pos < epoch,
                    dropwhile(
                        lambda x: (epoch - 1 > x.pos or x.pos > epoch), getattr(self, k)
                    ),
                )
                items = list(items)
                if items == []:
                    continue
                xs.append(epoch)
                avgs[k].append(np.mean([v for pos, v in items]))
            xs = sorted(set(xs))

        for k in avgs:
            if "val" in k:
                _type = "--"
            elif "test" in k:
                _type = ":"
            else:
                _type = "-"
            if len(avgs[k]) != len(xs):
                logger.info(
                    f"metric {k} was not fully recorded. Plotting final epochs using last recorded value"
                )
                avgs[k].extend([avgs[k][-1]] * (len(xs) - len(avgs[k])))
            ax.plot(
                xs,
                avgs[k],
                _type,
                label=k,
            )
        ax.grid(True)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Metrics")
        ax.set_title(
            kwargs.get("title", None), fontdict=kwargs.get("fontdict", {"size": 20})
        )
        if kwargs.get("log", False):
            ax.semilogy()
        plt.legend()
        if _show:
            plt.show()

    def report_avgs(self, epoch, return_avgs=True, end="\n"):
        avgs = {}
        for k in self.logged:
            avgs[k] = np.mean(
                [v for pos, v in getattr(self, k) if epoch - 1 <= pos < epoch]
            )
        self.report_metrics(epoch, end=end, **avgs)
        avgs = {f"epoch_{k}": v for k, v in avgs.items()}
        if self.wandb_logging:
            wandb.log({**avgs, "epoch": epoch})
        if self.mlflow_logging:
            mlflow.log_metrics(avgs, step=epoch)
        if return_avgs:
            return avgs

    def report_metrics(self, pos, **report):
        """Report training and validation metrics
        Required variables to be initialized before calling this function:
        1. start (time.time())
        2. n_epochs (int)

        Special kwargs:
        1. end - line ending after print (default newline)
        2. log - prefix info before printing summary

        Special argument:
        1. pos - position in training/testing process - float between 0 - n_epochs

        Usage:
        report_metrics(pos=1.42, train_loss=train_loss, validation_loss=validation_loss, ... )
        where each kwarg is a float
        """
        elapsed = time.time() - self.start
        end = report.pop("end", "\n")
        log = report.pop("log", "")
        log = log + ": " if log != "" else log
        elapsed = "  ({:.2f}s - {:.2f}s remaining)".format(
            time.time() - self.start, ((self.n_epochs - pos) / pos) * elapsed
        )
        current_iteration = f"EPOCH: {pos:.3f}  "
        if end == "\r":
            print(
                f"\r{log}{current_iteration}{info(report, self.precision)}{elapsed}",
                end="",
            )
        else:
            print(
                f"\r{log}{current_iteration}{info(report, self.precision)}{elapsed}",
                end=end,
            )

    def set_external_logging(self, project=None, hyper_parameters=None):
        if project is not None:
            if "/" in project:
                project, name = project.split("/")
            self.wandb_logging = True
            wandb.init(project=project, name=name, config=hyper_parameters)
        else:
            self.wandb_logging = False

        if mlflow and mlflow.active_run():
            self.mlflow_logging = True
        else:
            self.mlflow_logging = False

        self.train_step = 0
        self.validation_step = 0

    def finish_run(self, **kwargs):
        if not kwargs.get("do_not_finish_wandb", False):
            if self.wandb_logging:
                wandb.finish()


try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.progress import ProgressBarBase

    class LightningReport(ProgressBarBase):
        def __init__(
            self,
            epochs,
            print_every=None,
            print_total=None,
            precision=4,
            old_report=None,
        ):
            super().__init__()
            self.enable = True
            self.epoch_ix = 0
            _report = old_report.report if old_report is not None else None
            self.report = Report(epochs, precision, _report)
            if print_every is not None:
                self.print_every = print_every
            else:
                if print_total is None:
                    if epochs < 11:
                        self.print_every = 1
                else:
                    self.print_every = epochs // print_total

        def disable(self):
            self.enable = False

        def on_epoch_end(self, trainer, pl_module):
            self.epoch_ix += 1
            if self.epoch_ix % self.print_every == 0:
                self.report.report_avgs(self.epoch_ix)

        def prefix_dict_keys(self, prefix, _dict):
            _d = {}
            for k in _dict:
                _d[f"{prefix}_{k}"] = _dict[k]
            return _d

        def on_train_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        ):
            super().on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
            if isinstance(outputs, list):
                loss = float(trainer.progress_bar_dict["loss"])
                outputs = outputs[0][0]["extra"]
                outputs["loss"] = loss
            outputs = self.prefix_dict_keys("trn", outputs)
            self.px = self.epoch_ix + ((1 + batch_idx) / self.total_train_batches)
            self.report.record(self.px, end="\r", **outputs)

        def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
        ):
            super().on_train_batch_end(
                trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
            )
            loss = float(trainer.progress_bar_dict["loss"])
            self.px = self.epoch_ix + ((1 + batch_idx) / self.total_val_batches)
            outputs = self.prefix_dict_keys("val", outputs)
            self.report.record(self.px, end="\r", **outputs)

        def __getattr__(self, attr, **kwargs):
            return getattr(self.report, attr, **kwargs)

    __all__ += ["LightningReport", "pl"]
except Exception as e:
    ...


def moving_average(a, n=3):
    b = np.zeros_like(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    _n = len(b) - n
    b[-_n - 1 :] = ret[(n - 1) :] / n
    return b


def save_torch_model_weights_from(model, fpath, verbose=True):
    "from model to fpath"
    torch.save(model.state_dict(), fpath)
    if verbose:
        fsize = os.path.getsize(fpath) >> 20
        logger.opt(depth=1).log("INFO", f"Saved weights of size ~{fsize} MB to {fpath}")


def load_torch_model_weights_to(model, fpath, device=None, verbose=True):
    "to model from fpath"
    if not device:
        model.load_state_dict(torch.load(fpath))
    else:
        model.load_state_dict(torch.load(fpath, map_location=device))
    if verbose:
        logger.opt(depth=1).log("INFO", f"Loaded weights from {fpath} to given model")


def detach(i):
    if isinstance(i, dict):
        for k, v in i.items():
            i[k] = detach(v)
        return i
    elif isinstance(i, (list, tuple)):
        return [detach(j) for j in i]
    else:
        return i.cpu().detach()


def cat_with_padding(tensors, mode="constant", value=-100):
    """
    Concatenates a list of tensors with padding to make them compatible along 0th dimension.

    Args:
        tensors (list of torch.Tensor): List of tensors to be concatenated.
        mode (str, optional): The padding mode. Default is "constant".
        value (int, optional): The padding value for "constant" mode. Default is -100.

    Returns:
        torch.Tensor: Concatenated tensor with padding to match dimensions.

    Note:
        - All tensors should have the same dimension except for the stacking dimension.
        - Padding is added to match the dimensions of the largest tensor in the specified stacking dimension.
    """
    assert all(
        [tensors[0].ndim == t.ndim for t in tensors[1:]]
    ), "All tensors should have the same number of dimensions"
    sizes = [(t.size()[1:]) for t in tensors]
    full_size = [max(s) for s in list(zip(*sizes))]
    if full_size == []:
        # Given tensors are just scalars
        return torch.stack(tensors)

    def make_padding(current_size, target_size):
        o = ()
        for curr_d, targ_d in zip(current_size[::-1], target_size[::-1]):
            diff_d = targ_d - curr_d
            o = o + (0, diff_d)
        return o

    out = torch.cat(
        [
            F.pad(x, make_padding(x.size()[1:], full_size), mode=mode, value=value)
            for x in tensors
        ],
        dim=0,
    )
    return out


def clean_ipython_hist():
    # Code in this function mainly copied from IPython source
    if not "get_ipython" in globals():
        return
    ip = get_ipython()
    user_ns = ip.user_ns
    ip.displayhook.flush()
    pc = ip.displayhook.prompt_count + 1
    for n in range(1, pc):
        user_ns.pop("_i" + repr(n), None)
    user_ns.update(dict(_i="", _ii="", _iii=""))
    hm = ip.history_manager
    hm.input_hist_parsed[:] = [""] * pc
    hm.input_hist_raw[:] = [""] * pc
    hm._i = hm._ii = hm._iii = hm._i00 = ""


def clean_tb():
    # h/t Piotr Czapla
    if hasattr(sys, "last_traceback"):
        traceback.clear_frames(sys.last_traceback)
        delattr(sys, "last_traceback")
    if hasattr(sys, "last_type"):
        delattr(sys, "last_type")
    if hasattr(sys, "last_value"):
        delattr(sys, "last_value")


def clean_gpu_mem():
    clean_tb()
    clean_ipython_hist()
    gc.collect()
    torch.cuda.empty_cache()


def get_latest_checkpoint(directory, prefix="checkpoint-"):
    """Gets the latest checkpoint directory in the given directory.
    Args:
    directory: The directory to search for checkpoint directories.
    Returns:
    The path to the latest checkpoint directory, or None if no checkpoint directories
    were found.
    """
    import glob
    import os

    checkpoint_directories = sorted(
        glob.glob(os.path.join(directory, f"{prefix}*")),
        key=lambda x: int(x.split("/")[-1].replace(prefix, "")),
    )
    if not checkpoint_directories:
        return None
    return checkpoint_directories[-1]
