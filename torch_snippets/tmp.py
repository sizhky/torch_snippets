import os, glob
from pathlib import Path  # , _PosixFlavour, _WindowsFlavour
from fastcore.foundation import patch_to, L


def Glob(x, extns=None, silent=False):
    files = glob.glob(x + "/*") if "*" not in x else glob.glob(x)
    if extns:
        if isinstance(extns, str):
            extns = extns.split(",")
        files = [f for f in files if any([f.endswith(ext) for ext in extns])]

    # if not silent: logger.opt(depth=1).log('INFO', '{} files found at {}'.format(len(files), x))
    return files


class P(Path):
    # _flavour = _PosixFlavour() if os.name == "posix" else _WindowsFlavour()

    def __new__(cls, *pathsegments):
        return super().__new__(cls, *pathsegments)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )

        # try:
        #     return getattr(self, name)
        # except:
        #     raise AttributeError(
        #         f"'{self.__class__.__name__}' object has no attribute '{name}'"
        #     )

        # Fetch the stems of all files in the directory
        fs = {stem(f): f for f in self.ls()}

        if name in fs:
            # If the stem matches, return the corresponding filename
            return fs[name]
        else:
            # Raise AttributeError if the attribute is not found
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no attribute '{name}'"
            )


def stem(fpath):
    return P(fpath).stem


def stems(folder, silent=False):
    if isinstance(folder, (str, P)):
        return L([stem(str(x)) for x in Glob(folder, silent=silent)])
    if isinstance(folder, list):
        return L([stem(x) for x in folder])


P.stems = lambda self: stems(self.ls())
ls = lambda self: L(self.iterdir())
P.ls = ls
P.__repr__ = lambda self: f"» {self}"
P.__og_dir__ = P.__dir__


@patch_to(P)
def __dir__(self):
    items = []
    for f in ls(self):
        _f = stem(f)
        items.append(_f)
    return self.__og_dir__() + items


if __name__ == "__main__":
    x = P()
    print(x.resolve())
    print(x.misc)
