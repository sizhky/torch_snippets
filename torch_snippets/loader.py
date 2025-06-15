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
    # ---- Logging funcs ---- #
    "Info",
    "Warn",
    "Debug",
    "Excep",
    "reset_logger",
    "get_logger_level",
    "in_debug_mode",
    "debug_mode",
    # "display",
    "typedispatch",
    "defaultdict",
    "Counter",
    "dcopy",
    "patch_to",
    "split",
    "train_test_split",
    "init_plt",
    "init_cv2",
]

import os
import re
import sys
from builtins import print
from pathlib import Path

import numpy as np
import pandas as pd
import PIL
import tqdm
from fastcore.all import L, delegates, patch_to
# from fastcore.dispatch import typedispatch
from plum import dispatch as typedispatch
from PIL import Image

# from .bb_utils import *
# from .logger import *

from .bb_utils import (
    randint,
    BB,
    df2bbs,
    bbs2df,
    bbfy,
    jitter,
    enlarge_bbs,
    shrink_bbs,
    to_relative,
    to_absolute,
)
from .logger import (
    logger,
    Info,
    Warn,
    Debug,
    Excep,
    reset_logger,
    get_logger_level,
    in_debug_mode,
    debug_mode,
)

import datetime
import pdb
from typing import Tuple, Union

# Aliases
E = enumerate
pd.read_pqt = pd.read_parquet

from fastcore.foundation import coll_repr, is_array


@patch_to(L)
def _repr_pretty_(self, p, cycle):
    p.text(
        "..."
        if cycle
        else repr(self.items) if is_array(self.items) else coll_repr(self, 20)
    )


from collections import Counter, defaultdict
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
def choose(List, n=1, verbose=True):
    if n == 1:
        o = List[randint(len(List))]
    else:
        o = L([choose(List, verbose=verbose) for _ in range(n)])
    if verbose:
        Info(f"Chose `{o}` from input")
    return o


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


rand = lambda n=6: "".join(  # noqa: E731
    choose(
        list("1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM"),
        n=n,
        verbose=False,
    )
)


randint = lambda high: np.random.randint(high)  # noqa: E731 F811
randint = np.random.randint


def Tqdm(x, total=None, desc=None):
    total = len(x) if total is None else total
    return tqdm.tqdm(x, total=total, desc=desc)


from tqdm import trange

old_now = lambda: str(datetime.datetime.now())[:-10].replace(" ", "_")  # noqa: E731
now = lambda: f"{datetime.datetime.now():%Y%m%d-%H%M}"  # noqa: E731


def read(fname, mode=1):
    init_cv2()
    img = cv2.imread(str(fname), mode)
    if mode == 1:
        img = img[..., ::-1]  # BGR to RGB
    return img


def readPIL(fname, mode="RGB"):
    if mode.lower() == "bw":
        mode = "L"
    return Image.open(str(fname)).convert(mode.upper())


def crop_from_bb(im, bb, padding=None):
    if isinstance(bb, list):
        return [crop_from_bb(im, _bb, padding=padding) for _bb in bb]
    x, y, X, Y = bb
    px, py, pX, pY = padding
    if max(x, y, X, Y) < 1.5:
        h, w = im.shape[:2]
        x, y, X, Y = BB(bb).absolute((h, w))
    y = max(0, y - py)
    Y = min(h, Y + pY)
    x = max(0, x - px)
    X = min(w, X + pX)
    return im.copy()[y:Y, x:X]


def rect(im, bb, c=None, th=2):
    init_cv2()
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


def common(*items, silent=True):
    from .logger import logger

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
    from .logger import logger

    if not rev:
        o = set(sorted(set(a) - set(b)))
    else:
        o = set(sorted(set(b) - set(a)))
    if not silent:
        logger.opt(depth=1).log("INFO", f"{len(o)} items found to differ")
    return o


def puttext(im, string, org, scale=1, color=(255, 0, 0), thickness=2):
    init_cv2()
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


def is_in_notebook():
    try:
        import importlib

        # Test adapted from tqdm.autonotebook: https://github.com/tqdm/tqdm/blob/master/tqdm/autonotebook.py
        get_ipython = sys.modules["IPython"].get_ipython
        if "IPKernelApp" not in get_ipython().config:
            raise ImportError("console")

        return importlib.util.find_spec("IPython") is not None
    except (AttributeError, ImportError, KeyError):
        return False


def init_plt():
    import matplotlib.patheffects as path_effects
    import matplotlib.pyplot as plt

    plt.rcParams["axes.edgecolor"] = "black"
    globals().update(locals())


def init_cv2():
    import cv2

    globals().update(locals())


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
    grid: bool = False,
    save_path: str = None,
    text_sz: int = None,
    df: pd.DataFrame = None,
    pts=None,
    conns=None,
    interactive: bool = False,
    jitter: int = None,
    frame_count: int = 1,
    font_path=None,
    **kwargs,
):
    "show an image"
    from IPython.display import display, display_html

    init_plt()
    if hasattr(img, "__show__"):
        import inspect

        frame = inspect.currentframe()
        args, _, _, values = inspect.getargvalues(frame)
        all_args = {key: values[key] for key in args if key != "kwargs"}
        all_args.update(kwargs)  # Include additional kwargs
        img.__show__(**all_args)
        return

    try:
        if isinstance(img, (str, Path)):
            if str(img).startswith("s3://"):
                from .s3_loader import download_s3_file

                f = "____".join(str(img).split("/"))
                f = f"/tmp/{f}.jpeg"
                if not os.path.exists(f):
                    download_s3_file(img, f)
                img = f
            img = read(str(img), 1)
        try:
            import torch

            if isinstance(img, torch.Tensor):
                img = img.cpu().detach().numpy().copy()
        except ModuleNotFoundError:
            pass
        if isinstance(img, PIL.Image.Image):
            img = np.array(img)
    except Exception as e:
        Warn(e)

    if title is None:
        import inspect as I

        frame = I.currentframe()
        for _ in range(frame_count):
            frame = frame.f_back
        for var_name, var_val in frame.f_locals.items():
            if var_val is img and var_name != "_" and not var_name.strip("_").isdigit():
                title = var_name

    if isinstance(img, pd.Series):
        img = img.to_frame()
    if isinstance(img, pd.DataFrame):
        df = img
        max_rows = kwargs.pop("max_rows", 30)
        max_rows = 10000 if max_rows == -1 else max_rows
        if is_in_notebook():
            html_str = ""
            html_str += '<th style="text-align:center"><td style="vertical-align:top">'
            if title is not None:
                html_str += f'<h3 style="text-align: center;">{title}</h3>'
            html_str += (
                df.to_html(max_rows=max_rows)
                .replace("table", 'table style="display:inline"')
                .replace(' style="display:inline"', "")
            )
            html_str += "</td></th>"
            display_html(html_str, raw=True)
        else:
            o = df.to_markdown()
            print(o)
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

        text_col = kwargs.pop("text_col", "info" if "info" in df.columns else "text")
        if text_col == "ixs":
            texts = df.index.tolist()
        elif text_col is not None and text_col in df.columns:
            texts = df[text_col]
        if "color" in df.columns:
            bb_colors = df["color"].tolist()
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
        if jitter:
            bbs = [bb.jitter(jitter) for bb in bbs]
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
    if texts is not None or texts == "ixs":
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
        if font_path is not None and os.path.exists(font_path):
            from matplotlib import font_manager as fm

            font_properties = fm.FontProperties(fname=font_path)
        else:
            font_properties = None

        [
            puttext(
                ax,
                text.replace("$", "\$"),
                tuple(bbs[ix][:2]),
                size=text_sz,
                font_properties=font_properties,
            )
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


def puttext(
    ax, string, org, size=15, color=(255, 0, 0), thickness=2, font_properties=None
):
    init_plt()
    x, y = org
    va = "top" if y < 15 else "bottom"
    text = ax.text(
        x,
        y,
        str(string),
        color="red",
        ha="left",
        va=va,
        size=size,
        font_properties=font_properties,
    )
    text.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    )


def subplots(ims, nc=5, figsize=(5, 5), silent=True, **kwargs):
    init_plt()
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


uint = lambda im: (255 * im).astype(np.uint8)
Blank = lambda *sh: uint(np.ones(sh))


def pdfilter(df, column, condition, silent=True):
    from .logger import logger

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
        init_cv2()
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
    init_cv2()
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
            # yield batch, *rest_
            yield None
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
        assert list(vs).count(-1) == 1, "Only atmost one `-1` is allowed"
        vs = [v if v != -1 else -sum(vs) for v in vs]
    Info(vs)
    assert sum(vs) == 1, f"Split percentages should add to 1, received sum={sum(vs)}"
    np.random.seed(random_state)
    assignments = np.random.choice(ks, size=len(items), p=vs)
    o = {k: [] for k in ks}
    for item, assigned in zip(items, assignments):
        o[assigned].append(item)
    return o


def train_test_split(*args, **kwargs):
    # This is done mainly to save time and memory during imports
    from sklearn.model_selection import train_test_split as tts

    return tts(*args, **kwargs)


@patch_to(L)
def to_json(self):
    return list(self)
