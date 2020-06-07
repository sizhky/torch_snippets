'v0.15.2+0.75'
'''Todo - v0.15.3
- [x] Make Colab compatible
- [c] have a better keyword for transient log (`temp`)
- [ ] accumulate lists of tensors
- [x] convert to numpy if torch value is sent
- [x] use regexs for plotting metrics
- [x] add .plot_epochs()
- [x] moving average
- [x] plot can accept axis
'''

__all__ = ['torch','th','torchvision','T','nn','np','F','Dataset','DataLoader','optim','Report']

import torch, torchvision
import torch as th
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import time, numpy as np, matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
import re

metric = namedtuple('metric', 'pos,val'.split(','))
info = lambda report: '\t'.join([f'{k}: {v:.3f}' for k,v in report.items()])

def to_np(x):
    if isinstance(x, torch.Tensor): return x.detach().cpu().numpy()
    else:
        return x

class Report:
    def __init__(self, n_epochs):
        self.start = time.time()
        self.n_epochs = n_epochs
        self.completed_epochs = -1
        self.logged = []

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

    def plot(self, keys:[list,str]=None, smooth=0, ax=None):
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
            ax.plot(xs, ys, label=k)
        ax.grid(True)
        ax.set_xlabel('Epochs'); ax.set_ylabel('Metrics')
        ax.legend()
        if _show: plt.show()

    def history(self, k):
        return [v for _,v in getattr(self,k)]

    def plot_epochs(self, keys:list=None):
        avgs = defaultdict(list)
        keys = self.logged if keys is None else keys
        for epoch in range(self.n_epochs):
            for k in keys:
                avgs[k].append(np.mean([v for pos,v in getattr(self,k) if epoch-1<=pos<epoch]))
        for k in avgs:
            plt.plot(avgs[k], label=k)
        plt.grid(True)
        plt.xlabel('Epochs'); plt.ylabel('Metrics')
        plt.legend()
        plt.show()

    def report_avgs(self, epoch):
        avgs = {}
        for k in self.logged:
            avgs[k] = np.mean([v for pos,v in getattr(self,k) if epoch-1<=pos<epoch])
        self.report_metrics(epoch, **avgs)

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
            print(f'\r{log}{current_iteration}{info(report)}{elapsed}', end='')
        else:
            print(f'\r{log}{current_iteration}{info(report)}{elapsed}', end=end)

def moving_average(a, n=3) :
    b = np.zeros_like(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    _n = len(b) - n
    b[-_n-1:] = ret[(n-1):] / n
    return b