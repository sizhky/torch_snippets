__all__ = [
    "B",
    "Blank",
    "batchify",
    "C",
    "choose",
    "common",
    "crop_from_bb",
    "diff",
    "E",
    "flatten",
    "Image",
    "jitter",
    "L",
    "lzip",
    "line",
    "lines",
    "to_absolute",
    "to_relative",
    "enlarge_bbs",
    "shrink_bbs",
    "logger",
    "np",
    "now",
    "nunique",
    "os",
    "pad",
    "pd",
    "pdfilter",
    "pdb",
    "plt",
    "PIL",
    "print",
    "puttext",
    "randint",
    "rand",
    "re",
    "read",
    "readPIL",
    "rect",
    "resize",
    "rotate",
    "see",
    "set_logging_level",
    "show",
    "store_attr",
    "subplots",
    "sys",
    "toss",
    "track",
    "tqdm",
    "Tqdm",
    "trange",
    "unique",
    "uint",
    "write",
    "BB",
    "bbfy",
    "xywh2xyXY",
    "df2bbs",
    "bbs2df",
    "Info",
    "Warn",
    "Debug",
    "Excep",
    "reset_logger",
    "get_logger_level",
    "in_debug_mode",
    "debug_mode",
    "display",
    "typedispatch",
    "defaultdict",
    "Counter",
    "dcopy",
    "patch_to",
    "split",
]


from .logger import *
from .bb_utils import *
from pathlib import Path
from fastcore.dispatch import typedispatch
from fastcore.all import delegates, patch_to, L

import glob, numpy as np, pandas as pd, tqdm, os, sys, re
from IPython.display import display, display_html
import PIL
from PIL import Image

try:
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader

    __all__ += ["torch", "nn", "F", "Dataset", "DataLoader", "optim"]
except:
    Warn(
        "Unable to load torch and dependent libraries from torch-snippets. \n"
        "Functionalities might be limited. pip install lovely-tensors in case there are torch related errors"
    )

try:
    import lovely_tensors as lt

    lt.monkey_patch()
except:
    ...

try:
    from sklearn.model_selection import train_test_split

    __all__ += ["train_test_split"]
except:
    ...

import matplotlib  # ; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pdb, datetime
from typing import Union, Tuple

E = enumerate

try:
    import cv2

    __all__ += ["cv2"]
except:
    logger.warning("Skipping cv2 import")

import time
from collections import defaultdict, Counter
from copy import deepcopy as dcopy

from rich.progress import track as _track

track = lambda iterator, description="": _track(iterator, description=description)


old_line = lambda N=66: print("=" * N)


def line(string="", lw=66, upper=True, pad="\N{Box Drawings Double Horizontal}"):
    i = string.center(lw, pad)
    if upper:
        i = i.upper()
    print(i)


def lines(n=3, string="", **kwargs):
    assert n // 2 == (n - 1) // 2, "`n` should be odd"
    for _ in range(n // 2):
        line(**kwargs)
    line(string=string, **kwargs)
    for _ in range(n // 2):
        line(**kwargs)


def see(*X, N=66):
    list(map(lambda x: print("=" * N + "\n{}".format(x)), X)) + [print("=" * N)]


def flatten(lists):
    return [y for x in lists for y in x]


unique = lambda l: list(sorted(set(l)))
nunique = lambda l: len(set(l))


@typedispatch
def choose(List, n=1):
    if n == 1:
        return List[randint(len(List))]
    else:
        return L([choose(List) for _ in range(n)])


@typedispatch
def choose(i: dict, n=1):
    keys = list(i.keys())
    return choose(keys, n=n)


@typedispatch
def choose(i: set, n=1):
    i = list(i)
    return choose(i, n=n)


@typedispatch
def choose(i: pd.DataFrame, n=1):
    o = i.sample(n)
    if n == 1:
        return o.squeeze()
    return o


rand = lambda n=6: "".join(
    choose(list("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"), n=n)
)


randint = lambda high: np.random.randint(high)
randint = np.random.randint


def Tqdm(x, total=None, desc=None):
    total = len(x) if total is None else total
    return tqdm.tqdm(x, total=total, desc=desc)


from tqdm import trange

now = lambda: str(datetime.datetime.now())[:-10].replace(" ", "_")


def read(fname, mode=0):
    img = cv2.imread(str(fname), mode)
    if mode == 1:
        img = img[..., ::-1]  # BGR to RGB
    return img


def readPIL(fname, mode="RGB"):
    if mode.lower() == "bw":
        mode = "L"
    return Image.open(str(fname)).convert(mode.upper())


def crop_from_bb(im, bb):
    if isinstance(bb, list):
        return [crop_from_bb(im, _bb) for _bb in bb]
    x, y, X, Y = bb
    if max(x, y, X, Y) < 1.5:
        h, w = im.shape[:2]
        x, y, X, Y = BB(bb).absolute((h, w))
    return im.copy()[y:Y, x:X]


def rect(im, bb, c=None, th=2):
    c = "g" if c is None else c
    _d = {"r": (255, 0, 0), "g": (0, 255, 0), "b": (0, 0, 255), "y": (255, 0, 255)}
    c = _d[c] if isinstance(c, str) else c
    x, y, X, Y = bb
    cv2.rectangle(im, (x, y), (X, Y), c, th)


def B(im, th=180):
    "Binarize Image"
    return 255 * (im > th).astype(np.uint8)


def C(im):
    "make bw into 3 channels"
    if im.shape == 3:
        return im
    else:
        return np.repeat(im[..., None], 3, 2)


def common_old(a, b):
    """Wrapper around set intersection"""
    x = set(a).intersection(set(b))
    logger.opt(depth=1).log(
        "INFO",
        f"{len(x)} items found common from containers of {len(a)} and {len(b)} items respectively",
    )
    return set(sorted(x))


def common(*items, silent=True):
    """Wrapper around set intersection"""
    x = set(items[0])
    for item in items[1:]:
        x = set(item).intersection(x)
    lens = [str(len(i)) for i in items]
    if not silent:
        logger.opt(depth=1).log(
            "INFO",
            f"{len(x)} items found common from containers of {', '.join(lens)} items respectively",
        )
    return set(sorted(x))


def diff(a, b, rev=False, silent=False):
    if not rev:
        o = set(sorted(set(a) - set(b)))
    else:
        o = set(sorted(set(b) - set(a)))
    if not silent:
        logger.opt(depth=1).log("INFO", f"{len(o)} items found to differ")
    return o


def puttext(im, string, org, scale=1, color=(255, 0, 0), thickness=2):
    x, y = org
    org = x, int(y + 30 * scale)
    cv2.putText(im, str(string), org, cv2.FONT_HERSHEY_COMPLEX, scale, color, thickness)


def rotate(im, angle, pad=None, return_type=np.ndarray, bbs=None):
    pad = np.median(np.array(im)) if pad is None else pad
    pad = int(pad)
    if isinstance(im, np.ndarray):
        im = Image.fromarray(im)
    im = im.rotate(angle, expand=1, fillcolor=(pad, pad, pad))
    return np.array(im)


def _jitter(i):
    return i + np.random.randint(4)


@delegates(plt.imshow)
def show(
    img=None,
    ax=None,
    title=None,
    sz=None,
    bbs=None,
    confs=None,
    texts=None,
    bb_colors=None,
    cmap="gray",
    grid=False,
    save_path=None,
    text_sz=None,
    df=None,
    pts=None,
    conns=None,
    interactive=False,
    **kwargs,
):
    "show an image"

    try:
        if isinstance(img, (str, Path)):
            img = read(str(img), 1)
        if isinstance(img, torch.Tensor):
            img = img.cpu().detach().numpy().copy()
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)

    except Exception as e:
        print(e)
    if isinstance(img, pd.DataFrame):
        df = img
        html_str = ""
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        if title is not None:
            html_str += f'<h2 style="text-align: center;">{title}</h2>'
        max_rows = kwargs.pop("max_rows", 30)
        max_rows = 10000 if max_rows == -1 else max_rows
        html_str += df.to_html(max_rows=max_rows).replace(
            "table", 'table style="display:inline"'
        )
        html_str += "</td></th>"
        display_html(html_str, raw=True)
        return
    if not isinstance(img, np.ndarray):
        display(img)
        return

    if len(img.shape) == 3 and len(img) == 3:
        # this is likely a torch tensor
        img = img.transpose(1, 2, 0)
    img = np.copy(img)
    if img.max() == 255:
        img = img.astype(np.uint8)
    h, w = img.shape[:2]

    if interactive:
        from .interactive_show import ishow

        ishow(img, df=df)
        return

    if sz is None:
        if w < 50:
            sz = 1
        elif w < 150:
            sz = 2
        elif w < 300:
            sz = 5
        elif w < 600:
            sz = 10
        else:
            sz = 20
    if isinstance(sz, int):
        sz = (sz, sz)
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", sz))
        _show = True
    else:
        _show = False

    if df is not None:
        if isinstance(df, (str, Path)):
            df = str(df)
            df = pd.read_csv(df) if df.endswith("csv") else pd.read_parquet(df)
        try:
            text_col = kwargs.pop("text_col", "text")
            if text_col == "ixs":
                texts = df.index.tolist()
            else:
                texts = df[text_col]
        except:
            pass
        bbs = df2bbs(df)  # assumes df has 'x,y,X,Y' columns or a single 'bb' column
    kwargs.pop("text_col") if "text_col" in kwargs else ...
    if isinstance(texts, pd.core.series.Series):
        texts = texts.tolist()
    if confs:
        colors = [[255, 0, 0], [223, 111, 0], [191, 191, 0], [79, 159, 0], [0, 128, 0]]
        bb_colors = [colors[int(cnf * 5) - 1] for cnf in confs]
    if isinstance(bbs, np.ndarray):
        bbs = bbs.astype(np.uint16).tolist()
    if bbs is not None:
        if "th" in kwargs:
            th = kwargs.get("th")
            kwargs.pop("th")
        else:
            if w < 800:
                th = 2
            elif w < 1600:
                th = 3
            else:
                th = 4
        if hasattr(bbs, "shape"):
            if isinstance(bbs, torch.Tensor):
                bbs = bbs.cpu().detach().numpy()
            bbs = bbs.astype(np.uint32).tolist()
        if len(bbs) > 0:
            _x_ = np.array(bbs).max()
        else:
            raise ValueError("Trying to plot with 0 bounding boxes...")
        rel = True if _x_ < 1.5 else False
        if rel:
            bbs = [BB(bb).absolute((h, w)) for bb in bbs]
        bb_colors = (
            [[randint(255) for _ in range(3)] for _ in range(len(bbs))]
            if bb_colors == "random"
            else bb_colors
        )
        bb_colors = [bb_colors] * len(bbs) if isinstance(bb_colors, str) else bb_colors
        bb_colors = [None] * len(bbs) if bb_colors is None else bb_colors
        img = C(img) if len(img.shape) == 2 else img
        [rect(img, tuple(bb), c=bb_colors[ix], th=th) for ix, bb in enumerate(bbs)]
    text_sz = text_sz if text_sz else (max(sz) * 3 // 5)
    if texts is not None:
        if hasattr(texts, "shape"):
            if isinstance(texts, torch.Tensor):
                texts = texts.cpu().detach().numpy()
            texts = texts.tolist()
        if texts == "ixs":
            texts = [i for i in range(len(bbs))]
        if callable(texts):
            texts = [texts(bb) for bb in bbs]
        assert len(texts) == len(bbs), "Expecting as many texts as bounding boxes"
        texts = list(map(str, texts))
        texts = ["*" if len(t.strip()) == 0 else t for t in texts]
        [
            puttext(ax, text.replace("$", "\$"), tuple(bbs[ix][:2]), size=text_sz)
            for ix, text in enumerate(texts)
        ]
    if title:
        ax.set_title(title, fontdict=kwargs.pop("fontdict", None))
    if pts:
        pts = np.array(pts)
        if pts.max() < 1.1:
            pts = (pts * np.array([[w, h]])).astype(np.uint16).tolist()
        ax.scatter(*zip(*pts), c=kwargs.pop("pts_color", "red"))
    if conns is not None:
        for start_ix, end_ix, meta in conns:
            _x, _y = bbs[start_ix].xc, bbs[start_ix].yc
            _X, _Y = bbs[end_ix].xc, bbs[end_ix].yc
            _dx, _dy = _X - _x, _Y - _y
            _xc, _yc = (_X + _x) // 2, (_Y + _y) // 2
            plt.arrow(
                _jitter(_x),
                _jitter(_y),
                _jitter(_dx),
                _jitter(_dy),
                length_includes_head=True,
                color="cyan",
                head_width=4,
                head_length=4,
                width=meta * 2,
            )
            if kwargs.get("conn_text", True):
                puttext(ax, f"{meta:.2f}", (_xc, _yc), size=text_sz)
        kwargs.pop("conn_text")
    ax.imshow(img, cmap=cmap, **kwargs)

    if grid:
        ax.grid()
    else:
        ax.set_axis_off()

    if save_path:
        fig.savefig(save_path)
        return
    if _show:
        plt.show()


def puttext(ax, string, org, size=15, color=(255, 0, 0), thickness=2):
    x, y = org
    va = "top" if y < 15 else "bottom"
    text = ax.text(x, y, str(string), color="red", ha="left", va=va, size=size)
    text.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    )


def subplots(ims, nc=5, figsize=(5, 5), silent=True, **kwargs):
    if len(ims) == 0:
        return
    titles = kwargs.pop("titles", [None] * len(ims))
    if isinstance(titles, str):
        if titles == "ixs":
            titles = [str(i) for i in range(len(ims))]
        else:
            titles = titles.split(",")
    if len(ims) <= 5 and nc == 5:
        nc = len(ims)
    nr = (len(ims) // nc) if len(ims) % nc == 0 else (1 + len(ims) // nc)
    if not silent:
        logger.opt(depth=1).log(
            "INFO", f"plotting {len(ims)} images in a grid of {nr}x{nc} @ {figsize}"
        )
    figsize = kwargs.pop("sz", figsize)
    figsize = (figsize, figsize) if isinstance(figsize, int) else figsize
    fig, axes = plt.subplots(nr, nc, figsize=figsize)
    return_axes = kwargs.pop("return_axes", False)
    axes = axes.flat
    fig.suptitle(kwargs.pop("suptitle", ""))
    dfs = kwargs.pop("dfs", [None] * len(ims))
    bbss = kwargs.pop("bbss", [None] * len(ims))
    if "text_col" in kwargs:
        text_cols = [kwargs.pop("text_col")] * len(ims)
    else:
        text_cols = kwargs.pop("text_cols", [None] * len(ims))
    titles = titles.split(",") if isinstance(titles, str) else titles
    for ix, (im, ax) in enumerate(zip(ims, axes)):
        show(
            im,
            ax=ax,
            title=titles[ix],
            df=dfs[ix],
            bbs=bbss[ix],
            text_col=text_cols[ix],
            **kwargs,
        )
    blank = np.eye(100) + np.eye(100)[::-1]
    for ax in axes:
        show(blank, ax=ax)
    plt.tight_layout()
    if return_axes:
        return axes
    plt.show()


class L_old(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)):
            return list.__getitem__(self, keys)
        return L([self[k] for k in keys])

    def sample(self, n=1):
        return [self[randint(len(self))] for _ in range(n)]


uint = lambda im: (255 * im).astype(np.uint8)
Blank = lambda *sh: uint(np.ones(sh))


def pdfilter(df, column, condition, silent=True):
    if not callable(condition):
        if isinstance(condition, list):
            condition = lambda x: x in condition
        else:
            condition = lambda x: x == condition
    _df = df[df[column].map(condition)]
    if not silent:
        logger.opt(depth=1).log("DEBUG", f"Filtering {len(_df)} items out of {len(df)}")
    return _df


def pdsort(df, column, asc=True):
    df.sort_values(column, ascending=asc)


def set_logging_level(level):
    logger.remove()
    logger.add(sys.stderr, level=level)


def resize_old(im: np.ndarray, sz: Union[float, Tuple[int, int]]):
    h, w = im.shape[:2]
    if isinstance(sz, float):
        frac = sz
        H, W = [int(i * frac) for i in [h, w]]
    elif isinstance(sz, int):
        H, W = sz, sz
    elif isinstance(sz, tuple):
        if sz[0] == -1:
            _, W = sz
            f = W / w
            H = int(f * h)
        elif sz[1] == -1:
            H, _ = sz
            f = H / h
            W = int(f * w)
        else:
            H, W = sz
    return cv2.resize(im, (W, H))


def resize(
    im: Union[np.ndarray, PIL.Image.Image],
    sz: Union[float, Tuple[int, int], Tuple[str, Tuple[int, int]]],
):
    """Resize an image based on info from sz
    *Aspect ratio is preserved
    Examples:
        >>> im = np.random.rand(100,100)
        >>> _im = resize(im, 50)                    ; assert _im.shape == (50,50)
        >>> _im = resize(im, 0.5)                   ; assert _im.shape == (50,50)   #*
        >>> _im = resize(im, (50,200))              ; assert _im.shape == (50,200)
        >>> _im = resize(im, (0.5,2.0))             ; assert _im.shape == (50,200)
        >>> _im = resize(im, (0.5,200))             ; assert _im.shape == (50,200)

        >>> im = np.random.rand(50,100)
        >>> _im = resize(im, (-1, 200))             ; assert _im.shape == (100,200) #*
        >>> _im = resize(im, (100, -1))             ; assert _im.shape == (100,200) #*
        >>> _im = resize(im, ('at-least',(40,400))) ; assert _im.shape == (200,400) #*
        >>> _im = resize(im, ('at-least',(400,40))) ; assert _im.shape == (400,800) #*
        >>> _im = resize(im, ('at-most', (40,400))) ; assert _im.shape == (40,80)   #*
        >>> _im = resize(im, ('at-most', (400,40))) ; assert _im.shape == (20,40)   #*
    """
    if isinstance(im, PIL.Image.Image):
        im = np.array(im)
        to_pil = True
    else:
        to_pil = False
    h, w = im.shape[:2]
    if isinstance(sz, (tuple, list)) and isinstance(sz[0], str):
        signal, (H, W) = sz
        assert signal in "at-least,at-most".split(
            ","
        ), "Resize type must be one of `at-least` or `at-most`"
        if signal == "at-least":
            f = max(H / h, W / w)
        if signal == "at-most":
            f = min(H / h, W / w)
        H, W = [i * f for i in [h, w]]
    elif isinstance(sz, float):
        frac = sz
        H, W = [i * frac for i in [h, w]]
    elif isinstance(sz, int):
        H, W = sz, sz
    elif isinstance(sz, tuple):
        H, W = sz
        if H == -1:
            _, W = sz
            f = W / w
            H = f * h
        elif W == -1:
            H, _ = sz
            f = H / h
            W = f * w
        elif isinstance(H, float):
            H = H * h
        elif isinstance(W, float):
            W = W * h
    H, W = int(H), int(W)
    im = cv2.resize(im, (W, H))
    if to_pil:
        im = PIL.Image.fromarray(im)
    return im


def pad(im, sz, pad_value=255):
    h, w = im.shape[:2]
    IM = np.ones(sz) * pad_value
    IM[:h, :w] = im
    return IM


def xywh2xyXY(bbs):
    if len(bbs) == 4 and isinstance(bbs[0], int):
        x, y, w, h = bbs
        return BB(x, y, x + w, y + h)
    return [xywh2xyXY(bb) for bb in bbs]


def _store_attr(self, anno, **attrs):
    for n, v in attrs.items():
        if n in anno:
            v = anno[n](v)
        setattr(self, n, v)
        self.__stored_args__[n] = v


def store_attr(names=None, self=None, but=None, cast=False, **attrs):
    "Store params named in comma-separated `names` from calling context into attrs in `self`"
    fr = sys._getframe(1)
    args = fr.f_code.co_varnames[: fr.f_code.co_argcount]
    if self:
        args = ("self", *args)
    else:
        self = fr.f_locals[args[0]]
    if not hasattr(self, "__stored_args__"):
        self.__stored_args__ = {}
    anno = self.__class__.__init__.__annotations__ if cast else {}
    if attrs:
        return _store_attr(self, anno, **attrs)
    ns = re.split(", *", names) if names else args[1:]
    but = [] if not but else but
    _store_attr(self, anno, **{n: fr.f_locals[n] for n in ns if n not in but})


def makedir(x):
    os.makedirs(x, exist_ok=True)


def parent(fpath):
    out = "/".join(fpath.split("/")[:-1])
    if out == "":
        return "./"
    else:
        return out


def write(image, fpath):
    makedir(parent(fpath))
    cv2.imwrite(fpath, image)


def lzip(*x):
    return list(zip(*x))


@patch_to(L)
def first(self):
    if len(self) > 0:
        return self[0]
    else:
        return None


@patch_to(L)
def get(self, condition):
    sublist = self.filter(condition)
    if len(sublist) > 0:
        return sublist.first()
    else:
        return None


@patch_to(L)
def get_all(self, condition):
    sublist = self.filter(condition)
    return sublist


def batchify(items, *rest, batch_size=32):
    """
    Yield batches of `batch_size` items at a time from one or more sequences.

    Parameters:
        items (iterable): The main sequence of items to be batchified.
        *rest (iterable, optional): Additional sequences that should have the same length as `items`.
        batch_size (int, optional): The number of items per batch.

    Yields:
        tuple: A tuple containing batches of items from `items` and optionally from the additional sequences.
    """
    n = len(items)
    if len(rest) > 0:
        assert all(
            [n == len(_rest) for _rest in rest]
        ), "batchify can only work with equal length containers"
    head = 0
    while head < n:
        tail = head + batch_size
        batch = items[head:tail]
        if rest:
            rest_ = [list(_rest[head:tail]) for _rest in rest]
            yield batch, *rest_
        else:
            yield batch
        head = tail


def toss(frac: float, pass_=True, fail_=False):
    return [fail_, pass_][np.random.uniform() < frac]


def phasify(items, n_phases: int):
    """
    [Doc generated by AI]
    Distributes a list of items into multiple phases and returns a list of lists, each containing items from a specific phase.

    Parameters:
        items (list): A list of items to be distributed among phases.
        n_phases (int): The number of phases to divide the items into.

    Returns:
        list: A list of lists, where each sublist contains items assigned to a specific phase.

    Example:
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        n_phases = 3
        result = phasify(items, n_phases)
        # Output: [[1, 4, 7], [2, 5, 8], [3, 6, 9]]
    """
    iterators = defaultdict(L)
    [iterators[ix % n_phases].append(item) for ix, item in enumerate(items)]
    return L(iterators.values())


def split(items, splits, random_state=10):
    ks, vs = lzip(*splits.items())
    if any([v == -1 for v in vs]):
        assert list(vs).count(-1) == 1, f"Only atmost one `-1` is allowed"
        vs = [v if v != -1 else -sum(vs) for v in vs]
    Info(vs)
    assert sum(vs) == 1, f"Split percentages should add to 1, {sum(vs)=}"
    np.random.seed(random_state)
    assignments = np.random.choice(ks, size=len(items), p=vs)
    o = {k: [] for k in ks}
    for item, assigned in zip(items, assignments):
        o[assigned].append(item)
    return o


@patch_to(L)
def to_json(self):
    return list(self)
