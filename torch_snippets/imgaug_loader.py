# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/imgaug_loader.ipynb (unless otherwise specified).

__all__ = ['do', 'bw', 'rotate', 'pad', 'rescale', 'crop', 'imgaugbbs2bbs', 'bbs2imgaugbbs']

# Cell
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from .loader import BB, PIL, np


def do(img, bbs, aug, cval=255):
    bbs = bbs2imgaugbbs(bbs, img)
    im, bbs = aug(images=[img], bounding_boxes=[bbs])
    im, bbs = (im[0], imgaugbbs2bbs(bbs))
    return im, bbs


def bw(img, bbs):
    aug = iaa.Grayscale()
    return do(img, bbs, aug)


def rotate(img, bbs, angle, cval=255):
    aug = iaa.Rotate(angle, cval=cval, fit_output=True)
    return do(img, bbs, aug)


def pad(img, bbs, sz=None, deltas=None, cval=0):
    h, w = img.shape[:2]
    if sz:
        H, W = sz
        deltas = (H - h) // 2, (W - w) // 2, (H - h) // 2, (W - w) // 2

    aug = iaa.Pad(deltas, pad_cval=cval)
    return do(img, bbs, aug)


def rescale(im, bbs, sz):
    if isinstance(im, PIL.Image.Image):
        to_pil = True
        im = np.array(im)
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
    aug = iaa.Resize({"height": H, "width": W})
    im, bbs = do(im, bbs, aug)
    if to_pil:
        im = PIL.Image.fromarray(im)
    return im, bbs


def crop(img, bbs, deltas):
    aug = iaa.Crop(deltas)
    return do(img, bbs, aug)


def imgaugbbs2bbs(bbs):
    if bbs is None:
        return None
    return [
        BB([int(i) for i in (bb.x1, bb.y1, bb.x2, bb.y2)])
        for bb in bbs[0].bounding_boxes
    ]


def bbs2imgaugbbs(bbs, img):
    if bbs is None:
        return None
    return BoundingBoxesOnImage(
        [BoundingBox(x1=x, y1=y, x2=X, y2=Y) for x, y, X, Y in bbs], shape=img.shape
    )