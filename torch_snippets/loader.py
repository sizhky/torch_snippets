'V1.16'
__all__ = [
    'B','Blank','BB','bbfy','C','choose','common','crop_from_bb','cv2', 'dumpdill','df2bbs','diff','find',
    'flatten','fname','find','fname2','glob','Glob','Image','inspect','jitter', 'L',
    'line','loaddill','logger','extn', 'makedir', 'np', 'now','nunique','os','pad','pd','pdfilter','parent','Path','pdb',
    'plt','PIL','puttext','randint','rand','read','readlines','readPIL','rect','rename_batch','resize','see',
    'set_logging_level','show','stem','stems','subplots','sys','tqdm','Tqdm','Timer','unique','uint','writelines','zip_files','unzip_file','xywh2xyXY',
    'remove_duplicates', 'md5',
    'Info','Warn','Debug','Excep'
]

import cv2, glob, numpy as np, pandas as pd, tqdm, os, sys
import PIL
from PIL import Image
try:
    import torch
    import torch.nn as nn
    from torch import optim
    from torch.nn import functional as F
    from torch.utils.data import Dataset, DataLoader
    __all__ += ['torch','nn','F','Dataset','DataLoader','optim']
except: ...
import matplotlib#; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pdb, datetime, dill
from pathlib import Path
try:
    from loguru import logger
except:
    class Logger:
        def __init__(self): pass
        def info(self, message): print(f'INFO:\t{message}')
        def debug(self, message): print(f'DEBUG:\t{message}')
        def warning(self, message): print(f'WARNING:\t{message}')
        def exception(self, message): print(f'EXCEPTION:\t{message}')
    logger = Logger()
Info  = lambda x: logger.info(x)
Warn  = lambda x: logger.warning(x)
Debug = lambda x: logger.debug(x)
Excep = lambda x: logger.exception(x)

import time
class Timer:
    def __init__(self, N):
        'print elapsed time every iteration and print out remaining time'
        'assumes this timer is called exactly N times or less'
        self.start = time.time()
        self.N = N

    def __call__(self, ix, info=None):
        elapsed = time.time() - self.start
        print('\r{}\t{}/{} ({:.2f}s - {:.2f}s remaining)'.format(info, ix+1, self.N, elapsed, (self.N-ix)*(elapsed/(ix+1))), end='')

old_line = lambda N=66: print('='*N)
def line(string='', lw=66, upper=True, pad='='):
    i = string.center(lw, pad)
    if upper: i = i.upper()
    print(i)

def see(*X, N=66): list(map(lambda x: print('='*N+'\n{}'.format(x)), X))+[print('='*N)]
def flatten(lists): return [y for x in lists for y in  x]
unique = lambda l: list(sorted(set(l)))
nunique = lambda l: len(set(l))
def choose(List, n=1):
    if n == 1: return List[randint(len(List))]
    else:      return [choose(List) for _ in range(n)]

rand = lambda : ''.join(choose(list('1234567890qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'), n=6))
def find(item, List):
    '''Find an `item` in a `List`
    >>> find('abc', ['ijk','asdfs','dfsabcdsf','lmnop'])
    'dgsabcdsf'
    >>> find('file1', ['/tmp/file0.jpg', '/tmp/file0.png', '/tmp/file1.jpg', '/tmp/file1.png', '/tmp/file2.jpg', '/tmp/file2.png'])
    ['/tmp/file1.jpg', '/tmp/file1.png']
    '''
    filtered = [i for i in List if item in i]
    if len(filtered) == 1: return filtered[0]
    return filtered

def inspect(*arrays, **kwargs):
    '''
    shows shape, min, max and mean of an array/list/dict of oreys
    Usage:
    >>> inspect(arr1, arr2, arr3, [arr4,arr5,arr6], arr7, [arr8, arr9],...)
    where every `arr` is  assume to have a .shape, .min, .max and .mean methods
    '''
    depth = kwargs.get('depth', 0)
    names = kwargs.get('names',None)
    if names is not None:
        if ',' in names: names = names.split(',')
        assert len(names) == len(arrays), 'Give as many names as there are tensors to inspect'
    line()

    for ix, arr in enumerate(arrays):
        name = '\t'*depth
        name = name + f'{names[ix].upper().strip()}:\n' + name if names is not None else name
        name = name
        typ = type(arr).__name__

        if isinstance(arr, list):
            if arr == []:
                print('[]')
            else:
                print(f'{name}List Of {len(arr)} items')
                inspect(*arr[:5], depth=depth+1)
                if len(arr) > 5:
                    print('\t'*(depth+1)+f'... ... {len(arr) - 5} more items')

        elif isinstance(arr, dict):
            print(f'{name}Dict Of {len(arr)} items')
            for ix,(k,v) in enumerate(arr.items()):
                # print(f'\t'*(depth)+f' {k}'.upper())
                inspect(v, depth=depth+1, names=[k])
                if ix == 4: break
            if len(arr) > 5:
                print('\t'*(depth)+f'... ... {len(arr) - 5} more items')

        elif hasattr(arr, 'shape'):
            sh, m, M, dtype = arr.shape, arr.min(), arr.max(), arr.dtype
            try: me = arr.mean()
            except: me = arr.float().mean()
            print(f'{name}{typ}\tShape: {sh}\tMin: {m:.3f}\tMax: {M:.3f}\tMean: {me:.3f}\tdtype: {dtype}')
            line()
        else:
            ln = len(arr)
            print(f'{name}{typ} Length: {ln}')
            line()

randint = lambda high: np.random.randint(high)
def Tqdm(x, total=None, desc=None):
    total = len(x) if total is None else total
    return tqdm.tqdm(x, total=total, desc=desc)

now = lambda : str(datetime.datetime.now())[:-10].replace(' ', '_')

def read(fname, mode=0):
    img = cv2.imread(str(fname), mode)
    if mode == 1: img = img[...,::-1] # BGR to RGB
    return img

def readPIL(fname, mode='RGB'):
    if mode.lower() == 'bw': mode = 'L'
    return Image.open(str(fname)).convert(mode.upper())

def crop_from_bb(im, bb):
    x,y,X,Y = bb
    return im.copy()[y:Y,x:X]
def rect(im, bb, c=None, th=2):
    c = (0,255,0) if c is None else c
    x,y,X,Y = bb
    cv2.rectangle(im, (x,y), (X,Y), c, th)

def B(im, th=180):
    'Binarize Image'
    return 255*(im > th).astype(np.uint8)
def C(im):
    'make bw into 3 channels'
    if im.shape==3: return im
    else:
        return np.repeat(im[...,None], 3, 2)

makedir = lambda x: os.makedirs(x, exist_ok=True)
fname = lambda fpath: fpath.split('/')[-1]
fname2 = lambda fpath: stem(fpath.split('/')[-1])
def stem(fpath): return '.'.join(fname(fpath).split('.')[:-1])
def stems(folder):
    if isinstance(folder, str) : return [stem(x) for x in Glob(folder)]
    if isinstance(folder, list): return [stem(x) for x in folder]

def parent(fpath):
    out = '/'.join(fpath.split('/')[:-1])
    if out == '': return './'
    else:         return out
extn = lambda x: x.split('.')[-1]
def Glob(x, silent=False):
    files = glob.glob(x+'/*') if '*' not in x else glob.glob(x)
    if not silent: logger.info('{} files found at {}'.format(len(files), x))
    return files

def rename_batch(folder, func, debug=False, one_file=False):
    'V.V.Imp: Use debug=True first to confirm file name changes are as expected'
    if isinstance(folder, str): folder = Glob(folder)
    sources = []
    destins = []
    log_file = f'moved_files_{now()}.log'
    for f in folder:
        source = f
        destin = func(f)
        if source == destin: continue
        if debug:
            logger.debug(f'moving `{source}` --> `{destin}`')
        else:
            # !mv {source.replace(' ','\ ')} {destin.replace(' ','\ ')}
            logger.info(f'moving `{source}` --> `{destin}`')
            os.rename(source, destin)
        # !echo {source.replace(' ','\ ')} --\> {destin.replace(' ','\ ')} >> {logfile}
        if one_file: break
def common(a, b):
    """Wrapper around set intersection"""
    x = list(set(a).intersection(set(b)))
    logger.info(f'{len(x)} items found common from containers of {len(a)} and {len(b)} items respectively')
    return sorted(x)
def diff(a, b, rev=False):
    if not rev:
        o = list(sorted(set(a) - set(b)))
        logger.info(f'{len(o)} items found to differ')
        return o
    else:
        o = list(sorted(set(b) - set(a)))
        logger.info(f'{len(o)} items found to differ')
        return o
def puttext(im, string, org, scale=1, color=(255,0,0), thickness=2):
    x,y = org
    org = x, int(y+30*scale)
    cv2.putText(im, str(string), org, cv2.FONT_HERSHEY_COMPLEX, scale, color, thickness)

def show(img=None, ax=None, title=None, sz=None, bbs=None, confs=None,
         texts=None, bb_colors=None, cmap='gray', grid=False,
         save_path=None, text_sz=15, df=None, **kwargs):
    'show an image'
    try:
        if isinstance(img, torch.Tensor): img = img.cpu().detach().numpy().copy()
        if isinstance(img, PIL.Image.Image): img = np.array(img)
    except: ...
    if len(img.shape) == 3 and len(img) == 3:
        # this is likely a torch tensor
        img = img.transpose(1,2,0)
    img = np.copy(img)
    if img.max() == 255: img = img.astype(np.uint8)
    h, w = img.shape[:2]
    if sz is None:
        if w<50: sz=1
        elif w<150: sz=2
        elif w<300: sz=5
        elif w<600: sz=10
        else: sz=20
    if isinstance(sz, int):
        sz = (sz, sz)
    if ax is None:
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', sz))
        _show = True
    else: _show = False

    if df is not None:
        try: texts = df.text
        except:
            pass
        bbs = df2bbs(df) # assumes df has 'x,y,X,Y' columns
    if isinstance(texts, pd.core.series.Series): texts = texts.tolist()
    if texts is not None:
        if hasattr(texts, 'shape'):
            if isinstance(texts, torch.Tensor): texts = texts.cpu().detach().numpy()
            texts = texts.tolist()
        if texts is 'ixs': texts = [i for i in range(len(bbs))]
        if callable(texts): texts = [texts(bb) for bb in bbs]
        assert len(texts) == len(bbs), 'Expecting as many texts as bounding boxes'
        texts = list(map(str, texts))
        texts = ['*' if len(t.strip())==0 else t for t in texts]
        [puttext(ax, text.replace('$','\$'), tuple(bbs[ix][:2]), size=text_sz) for ix,text in enumerate(texts)]
    if confs:
        colors = [[255, 0, 0], [223, 111, 0], [191, 191, 0], [79, 159, 0], [0, 128, 0]]
        bb_colors = [colors[ int(cnf*5)-1 ] for cnf in confs]
    if isinstance(bbs, np.ndarray): bbs = bbs.astype(np.uint16).tolist()
    if bbs is not None:
        if 'th' in kwargs:
            th = kwargs.get('th')
            kwargs.pop('th')
        else:
            if w<800: th=2
            elif w<1600: th=3
            else: th=4
        if hasattr(bbs, 'shape'):
            if isinstance(bbs, torch.Tensor): bbs = bbs.cpu().detach().numpy()
            bbs = bbs.astype(np.uint32).tolist()

        bb_colors = [[randint(255) for _ in range(3)] for _ in range(len(bbs))] if bb_colors is 'random' else bb_colors
        bb_colors = [None]*len(bbs) if bb_colors is None else bb_colors
        img = C(img) if len(img.shape) == 2 else img
        [rect(img, tuple(bb), c=bb_colors[ix], th=th) for ix,bb in enumerate(bbs)]

    if title: ax.set_title(title, fontdict=kwargs.pop('fontdict', None))
    ax.imshow(img, cmap=cmap, **kwargs)

    if not grid: ax.set_axis_off()
    if save_path:
        fig.savefig(save_path)
        return
    if _show: plt.show()

def puttext(ax, string, org, size=15, color=(255,0,0), thickness=2):
    x,y = org
    va = 'top' if y < 15 else 'bottom'
    text = ax.text(x, y, str(string), color='red', ha='left', va=va, size=size)
    text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='white'),
                           path_effects.Normal()])

def dumpdill(obj, fpath, silent=False):
    os.makedirs(parent(fpath), exist_ok=True)
    with open(fpath, 'wb') as f:
        dill.dump(obj, f)
    if not silent: logger.info('Dumped object @ {}'.format(fpath))

def loaddill(fpath):
    with open(fpath, 'rb') as f:
        obj = dill.load(f)
    return obj

class BB:
    def __init__(self, *bb):
        rel = False
        # assert len(bb) == 4, 'expecting a list/tuple of 4 values respectively for (x,y,X,Y)'
        if len(bb) == 4: x,y,X,Y = bb
        elif len(bb) == 1: (x,y,X,Y), = bb
        if not rel: x,y,X,Y = map(lambda i: int(round(i)), (x,y,X,Y))
        self.bb = x,y,X,Y
        self.x, self.y, self.X, self.Y = x,y,X,Y
        self.xc, self.yc = (self.x+self.X)/2, (self.y+self.Y)/2
        self.h = Y-y
        self.w = X-x
        self.area = self.h * self.w

    def __getitem__(self, i): return self.bb[i]
    def __repr__(self): return self.bb.__repr__()
    def __len__(self): return 4
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y and self.X == other.X and self.Y == other.Y
    def __hash__(self): return hash(tuple(self))
    def remap(self, og_dim:('h','w'), new_dim:('H','W')):
        h,w = og_dim
        H,W = new_dim
        sf_x = H/h
        sf_y = W/w
        return BB(round(sf_x*self.x), round(sf_y*self.y), round(sf_x*self.X), round(sf_y*self.Y))
    def relative(self, dim:('h','w')):
        h, w = dim
        return BB(self.x/w, self.y/h, self.X/w, self.Y/h, rel=True)
    def local_to(self, _bb):
        x,y,X,Y = self
        a,b,A,B = _bb
        return BB(x-a, y-b, X-a, Y-b)
    def jitter(self, noise):
            return BB([i+(randint(2*noise)-noise) for i in self])
    def add_padding(self, *pad):
        if len(pad) == 4: _x,_y,_X,_Y = pad
        else:
            pad, = pad
            _x,_y,_X,_Y = pad,pad,pad,pad
        x , y, X, Y = self.bb
        return max(0,x-_x), max(0,y-_y), X+_x,Y+_y

def subplots(ims, nc=5, figsize=(5,5), **kwargs):
    if len(ims) == 0: return
    titles = kwargs.pop('titles',[None]*len(ims))
    if isinstance(titles, str):
        titles = titles.split(',')
    if len(ims) <= 3: nc=len(ims)
    nr = (len(ims)//nc) if len(ims)%nc==0 else (1+len(ims)//nc)
    logger.info(f'plotting {len(ims)} images in a grid of {nr}x{nc} @ {figsize}')
    figsize = kwargs.pop('sz', figsize)
    figsize = (figsize,figsize) if isinstance(figsize, int) else figsize
    fig, axes = plt.subplots(nr,nc,figsize=figsize)
    axes = axes.flat
    fig.suptitle(kwargs.pop('suptitle',''))
    titles = titles.split(',') if isinstance(titles, str) else titles
    for ix,(im,ax) in enumerate(zip(ims,axes)):
        show(im, ax=ax, title=titles[ix], **kwargs)
    blank = (np.eye(100) + np.eye(100)[::-1])
    for ax in axes: show(blank, ax=ax)
    plt.tight_layout()
    plt.show()

df2bbs = lambda df: [BB(bb) for bb in df[list('xyXY')].values.tolist()]
def bbfy(bbs): return [BB(bb) for bb in bbs]
def jitter(bbs, noise):
    return [BB(bb).jitter(noise) for bb in bbs]

class L(list):
    def __getitem__(self, keys):
        if isinstance(keys, (int, slice)): return list.__getitem__(self, keys)
        return L([self[k] for k in keys])
    def sample(self, n=1):
        return [self[randint(len(self))] for _ in range(n)]

uint = lambda im: (255*im).astype(np.uint8)
Blank = lambda *sh: uint(np.ones(sh))

def pdfilter(df, column, condition):
    _df = df[df[column].map(condition)]
    logger.debug(f'Filtering {len(_df)} items out of {len(df)}')
    return _df
def pdsort(df, column, asc=True):
    df.sort_values(column, ascending=asc)

def set_logging_level(level):
    logger.remove()
    logger.add(sys.stderr, level=level)

def resize_old(im:np.ndarray, sz:[float,('H','W')]):
    h,w = im.shape[:2]
    if isinstance(sz, float):
        frac = sz
        H,W = [int(i*frac) for i in [h,w]]
    elif isinstance(sz, int):
        H,W = sz,sz
    elif isinstance(sz, tuple):
        if sz[0] == -1:
            _,W = sz
            f = W/w
            H = int(f*h)
        elif sz[1] == -1:
            H,_ = sz
            f = H/h
            W = int(f*w)
        else:
            H,W = sz
    return cv2.resize(im, (W,H))

def readlines(fpath, silent=False):
    with open(fpath, 'r') as f:
        lines = f.read().split('\n')
        lines = [l.strip() for l in lines if l.strip()!='']
        if not silent: Info(f'loaded {len(lines)} lines')
        return lines

def resize(im:np.ndarray, sz:[float,('H','W'),(str,('H','W'))]):
    '''Resize an image based on info from sz
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
   '''
    h,w = im.shape[:2]
    if isinstance(sz, (tuple,list)) and isinstance(sz[0], str):
        signal, (H,W) = sz
        assert signal in 'at-least,at-most'.split(','),\
            'Resize type must be one of `at-least` or `at-most`'
        if signal == 'at-least':
            f = max(H/h,W/w)
        if signal == 'at-most':
            f = min(H/h,W/w)
        H,W = [i*f for i in [h,w]]
    elif isinstance(sz, float):
        frac = sz
        H,W = [i*frac for i in [h,w]]
    elif isinstance(sz, int):
        H,W = sz,sz
    elif isinstance(sz, tuple):
        H,W = sz
        if H == -1:
            _,W = sz
            f = W/w
            H = f*h
        elif W == -1:
            H,_ = sz
            f = H/h
            W = f*w
        elif isinstance(H, float): H = H*h
        elif isinstance(W, float): W = W*h
    H,W = int(H),int(W)
    return cv2.resize(im, (W,H))

def pad(im, sz, pad_value=255):
    h,w = im.shape[:2]
    IM = np.ones(sz)*pad_value
    IM[:h,:w] = im
    return IM

def writelines(lines, file):
    makedir(parent(file))
    failed = []
    with open(file, 'w') as f:
        for line in lines:
            try: f.write(f'{line}\n')
            except: failed.append(line)
    if failed!=[]:
        logger.info(f'Failed to write {len(failed)} lines out of {len(lines)}')
        return failed

def zip_files(list_of_files, dest):
    import zipfile
    Info(f'Zipping {len(list_of_files)} files to {dest}...')
    with zipfile.ZipFile(dest, 'w') as zipMe:
        for file in Tqdm(list_of_files):
            zipMe.write(file, compress_type=zipfile.ZIP_DEFLATED)

def unzip_file(file, dest):
    import zipfile
    with zipfile.ZipFile(file, 'r') as zip_ref:
        zip_ref.extractall(dest)

def md5(fname):
    hash_md5 = hashlib.md5()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def remove_duplicates(files):
    hashes = [md5(f) for f in files]
    df = pd.DataFrame({'f':files, 'h': hashes})
    x = df.drop_duplicates('h')
    y = diff(files, x.f)
    for i in y:
        os.rename(i, './x')
    # !rm ./x
    return


def xywh2xyXY(bbs):
    if len(bbs) == 4 and isinstance(bbs[0], int):
        x,y,w,h = bbs
        return BBox(x,y,x+w,y+h)
    return [xywh2xyXY(bb) for bb in bbs]
