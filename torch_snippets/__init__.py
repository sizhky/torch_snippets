__version__ = "0.556"
from .logger import *
from .loader import *
from .paths import *
from .markup import *
from .markdown import *

# from .inspector import *
from .load_defaults import *
from .pdf_loader import PDF
from .markup2 import AD
from .registry import *
from .ipython import *
from .decorators import *
from .misc import *
from .dates import *
from .s3_loader import *
from .zen import *

# Optional lazy-load safe placeholders for linters
torch = nn = F = th = T = transforms = torchvision = np = Dataset = DataLoader = (
    optim
) = Report = Reshape = Permute = device = save_torch_model_weights_from = (
    load_torch_model_weights_to
) = detach = cat_with_padding = None

def init_torch():
    try:
        from .torch_loader import (
            torch,
            th,
            torchvision,
            T,
            transforms,
            nn,
            np,
            F,
            Dataset,
            DataLoader,
            optim,
            Report,
            Reshape,
            Permute,
            device,
            save_torch_model_weights_from,
            load_torch_model_weights_to,
            detach,
            cat_with_padding,
        )

        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Unable to install torch dependencies. Please install them using `pip install torch-snippets[torch]`"
        )

    import builtins

    # Define a dictionary mapping names to imported objects
    modules = {
        "torch": torch,
        "th": th,
        "torchvision": torchvision,
        "T": T,
        "transforms": transforms,
        "nn": nn,
        "np": np,
        "F": F,
        "Dataset": Dataset,
        "DataLoader": DataLoader,
        "optim": optim,
        "Report": Report,
        "Reshape": Reshape,
        "Permute": Permute,
        "device": device,
        "save_torch_model_weights_from": save_torch_model_weights_from,
        "load_torch_model_weights_to": load_torch_model_weights_to,
        "detach": detach,
        "cat_with_padding": cat_with_padding,
    }

    # Use a loop to set attributes in builtins
    for name, module in modules.items():
        setattr(builtins, name, module)
