'v0.14.0'
__all__ = ['torch','nn','np','F','Dataset','DataLoader','optim','Report']

import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import time, numpy as np, matplotlib.pyplot as plt
from collections import namedtuple

metric = namedtuple('metric', 'pos,val'.split(','))
info = lambda report: '\t'.join([f'{k}: {v:.3f}' for k,v in report.items()])

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
                getattr(self, k).append(metric(pos, v))
            else:
                setattr(self, k, [])
                getattr(self, k).append(metric(pos, v))
                self.logged.append(k)
        self.report_metrics(pos, **metrics)

    def plot(self, keys:list=None):
        keys = self.logged if keys is None else keys
        for k in keys:
            xs, ys = list(zip(*getattr(self,k)))
            plt.plot(xs, ys, label=k)
        plt.grid(True)
        plt.xlabel('Epochs'); plt.ylabel('Metrics')
        plt.legend()
        plt.show()

    def history(self, k):
        return [v for _,v in getattr(self,k)]

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
        print(log + current_iteration + info(report) + elapsed, end=end)
