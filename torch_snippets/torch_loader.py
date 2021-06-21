'v0.15.41'
'''Todo - v0.15.5
- [ ] accumulate lists of tensors
- [ ] make plot_epochs even faster
'''

__all__ = ['torch','th','torchvision','T','transforms','nn','np','F','Dataset','DataLoader','optim','Report','Reshape','Permute','device','save_torch_model_weights_from','load_torch_model_weights_to']

import torch, torchvision
import torch as th
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import time, numpy as np, matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
import re
from loguru import logger
from itertools import dropwhile, takewhile
from .loader import makedir, os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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

metric = namedtuple('metric', 'pos,val'.split(','))
info = lambda report, precision: '\t'.join([f'{k}: {v:.{precision}f}' for k,v in report.items()])

def to_np(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    else:
        return x

class Report:
    def __init__(self, n_epochs, precision=3, old_report=None):
        self.start = time.time()
        self.n_epochs = n_epochs
        self.precision = precision
        self.completed_epochs = -1
        self.logged = []
        if old_report:
            self.prepend(old_report)

    def prepend(self, old_report):
        for k in old_report.logged:
            self.logged.append(k)
            last = getattr(old_report, k)[-1].pos
            setattr(self, k, [])
            for m in getattr(old_report, k):
                getattr(self, k).append(metric(m.pos - last, to_np(m.val)))

    def record(self, pos, **metrics):
        for k,v in metrics.items():
            if k in ['end','pos']: continue
            if hasattr(self, k):
                getattr(self, k).append(metric(pos, to_np(v)))
            else:
                setattr(self, k, [])
                getattr(self, k).append(metric(pos, to_np(v)))
                self.logged.append(k)
        self.report_metrics(pos, **metrics)

    def plot(self, keys:[list,str]=None, smooth=0, ax=None, **kwargs):
        _show = True if ax is None else False
        if ax is None:
            sz = 8,6
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', sz))

        keys = self.logged if keys is None else keys
        if isinstance(keys, str):
            key_pattern = keys
            keys = [key for key in self.logged if re.search(key_pattern, key)]

        for k in keys:
            xs, ys = list(zip(*getattr(self,k)))
            if smooth: ys = moving_average(np.array(ys), smooth)

            if 'val' in k   : _type = '--'
            elif 'test' in k: _type = ':'
            else            : _type = '-'

            ax.plot(xs, ys, _type, label=k)
        ax.grid(True)
        ax.set_xlabel('Epochs'); ax.set_ylabel('Metrics')
        ax.set_title(kwargs.get('title', None), fontdict=kwargs.get('fontdict', {'size': 20}))
        if kwargs.get('log', False): ax.semilogy()
        ax.legend()
        if _show: plt.show()

    def history(self, k):
        return [v for _,v in getattr(self,k)]

    def plot_epochs(self, keys:list=None, ax=None, **kwargs):
        _show = True if ax is None else False
        if ax is None:
            sz = 8,6
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', sz))
        avgs = defaultdict(list)
        keys = self.logged if keys is None else keys

        from tqdm import trange
        if isinstance(keys, str):
            key_pattern = keys
            keys = [key for key in self.logged if re.search(key_pattern, key)]
        xs = []
        for epoch in trange(-100, self.n_epochs+1):
            for k in keys:
                items = takewhile(lambda x: epoch-1<=x.pos<epoch,
                    dropwhile(lambda x: (epoch-1>x.pos or x.pos>epoch), getattr(self,k)))
                items = list(items)
                if items == []: continue
                xs.append(epoch)
                avgs[k].append(np.mean([v for pos,v in items]))
            xs = sorted(set(xs))

        for k in avgs:

            if 'val' in k   : _type = '--'
            elif 'test' in k: _type = ':'
            else            : _type = '-'
            if len(avgs[k]) != len(xs):
                logger.info(f'metric {k} was not fully recorded. Plotting final epochs using last recorded value')
                avgs[k].extend([avgs[k][-1]]*(len(xs)-len(avgs[k])))
            ax.plot(xs, avgs[k], _type, label=k,)
        ax.grid(True)
        ax.set_xlabel('Epochs'); ax.set_ylabel('Metrics')
        ax.set_title(kwargs.get('title', None), fontdict=kwargs.get('fontdict', {'size': 20}))
        if kwargs.get('log', False): ax.semilogy()
        plt.legend()
        if _show: plt.show()

    def report_avgs(self, epoch, return_avgs=False):
        avgs = {}
        for k in self.logged:
            avgs[k] = np.mean([v for pos,v in getattr(self,k) if epoch-1<=pos<epoch])
        self.report_metrics(epoch, **avgs)
        if return_avgs: return avgs

    def report_metrics(self, pos, **report):
        '''Report training and validation metrics
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
        '''
        elapsed = time.time() - self.start
        end = report.pop('end','\n')
        log = report.pop('log', ''); log = log+': ' if log!='' else log
        elapsed = '\t({:.2f}s - {:.2f}s remaining)'.format(time.time() - self.start, ((self.n_epochs-pos)/pos)*elapsed)
        current_iteration = f'EPOCH: {pos:.3f}\t'
        if end == '\r':
            print(f'\r{log}{current_iteration}{info(report, self.precision)}{elapsed}', end='')
        else:
            print(f'\r{log}{current_iteration}{info(report, self.precision)}{elapsed}', end=end)

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks.progress import ProgressBarBase
    class LightningReport(ProgressBarBase):
        def __init__(self, epochs, print_every=None, print_total=None, precision=4, old_report=None):
            super().__init__()
            self.enable = True
            self.epoch_ix = 0
            _report = old_report.report if old_report is not None else None
            self.report = Report(epochs, precision, _report)
            if print_every is not None:
                self.print_every = print_every
            else:
                if print_total is None:
                    if epochs < 11: self.print_every = 1
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
                _d[f'{prefix}_{k}'] = _dict[k]
            return _d

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if isinstance(outputs, list):
                loss = float(trainer.progress_bar_dict["loss"])
                outputs = outputs[0][0]['extra']
                outputs['loss'] = loss
            outputs = self.prefix_dict_keys('trn', outputs)
            self.px = self.epoch_ix + ((1+batch_idx)/self.total_train_batches)
            self.report.record(self.px, end='\r', **outputs)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
            super().on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            loss = float(trainer.progress_bar_dict["loss"])
            self.px = self.epoch_ix + ((1+batch_idx)/self.total_val_batches)
            outputs = self.prefix_dict_keys('val', outputs)
            self.report.record(self.px, end='\r', **outputs)

        def __getattr__(self, attr, **kwargs):
            return getattr(self.report, attr, **kwargs)

    __all__ += ['LightningReport', 'pl']
except:
    logger.warning('Not importing Lightning Report')

def moving_average(a, n=3) :
    b = np.zeros_like(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    _n = len(b) - n
    b[-_n-1:] = ret[(n-1):] / n
    return b


def save_torch_model_weights_from(model, fpath):
    'from model to fpath'
    torch.save(model.state_dict(), fpath)
    fsize = os.path.getsize(fpath) >> 20
    logger.opt(depth=1).log('INFO', f'Saved weights of size ~{fsize} MB to {fpath}')

def load_torch_model_weights_to(model, fpath):
    'to model from fpath'
    model.load_state_dict(torch.load(fpath))
    logger.opt(depth=1).log('INFO', f'Loaded weights from {fpath} to given model')
