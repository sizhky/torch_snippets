## torch utilities for recording metrics and plotting
```python
n_epochs = 5
log = Report(n_epochs)

for epoch in range(n_epochs):
    train_epoch_losses, train_epoch_accuracies = [], []
    N = len(trn_dl)
    for ix, batch in enumerate(iter(trn_dl)):
        ...
        pos = (epoch + (ix+1)/N) # a float between 0 - n_epochs
        # give any number of kwargs that need to be reported and stored.
        # args should be float
        log.record(pos=pos, train_acc=np.mean(is_correct), train_loss=batch_loss, end='\r') # impersistent log

    N = len(val_dl)
    for ix, batch in enumerate(iter(val_dl)):
        ...
        pos = (epoch + (ix+1)/N) # a float between 0 - n_epochs
        log.record(pos=pos, val_loss=batch_loss, end='\r') # impersistent log
    log.report_avgs(epoch+1) # persist the report

```

![](assets/demo.gif)

```python
log.plot() # plot everything that has been recorded
```
![](assets/avgs.png)

## Install
pip install torch_snippets

## Usage
```python
from torch_snippets import *
log = Record(n_epochs) # number of epochs to be trained
log.record(pos, **kwargs) # where each kwarg is a float and 
# pos is the current position in training a float between 0 and n_epochs
log.report_avgs(epoch+1) # avgs of all metrics logged between `epoch` and `epoch+1`
```
#### Note
use `log.record(..., end='\r')` for a temporary log which will be overwritten
