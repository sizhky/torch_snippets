__version__ = "0.540"
from .logger import *
from .loader import *
from .paths import *
from .markup import *

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


def init_torch():
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

    try:
        import lovely_tensors

        lovely_tensors.monkey_patch()
    except ModuleNotFoundError:
        raise ModuleNotFoundError(
            "Unable to import `lovely-tensors`. `pip install lovely-tensors` for a prettier experience"
        )

    globals().update(locals())
