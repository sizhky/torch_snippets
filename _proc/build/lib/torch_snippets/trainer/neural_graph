from .hooks import attach_hooks
from ..markup2 import AD


class ModelIO:
    def __init__(self):
        self.layers = {}

    def save(self, module, input, kwargs, output, layer_name=""):
        info = AD(input=input, kwargs=kwargs, output=output)
        self.layers[layer_name] = info
