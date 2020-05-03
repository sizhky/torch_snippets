import matplotlib.pyplot as plt
import seaborn as sns

def plot(*args, **kwargs):
    item = kwargs.pop('i',plt)
    for a in args:
        if isinstance(a, dict)                                 : o = plot(i=item, **a)
    for k,v in kwargs.items():
        if isinstance(v, dict)                                 : o = getattr(item, k)(**v)
        elif isinstance(v, (list, tuple))                      :
            if not any([isinstance(item, dict) for item in v]) : o = getattr(item, k)(*v)
            else:
                _v, _kw = [i for i in v if not isinstance(i, dict)], [i for i in v if isinstance(i, dict)][0]
                o = getattr(item, k)(*_v, **_kw)
        else                                                   : o = getattr(item, k)(v)
    return o

def snsplot(*args, **kwargs):
    item = sns
    return plot(i=item, *args, **kwargs)