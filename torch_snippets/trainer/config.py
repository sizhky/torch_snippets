# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/config.ipynb.

# %% auto 0
__all__ = ["DeepLearningConfig", "GenericConfig"]

# %% ../../nbs/config.ipynb 1
from ..registry import parse
from .. import store_attr, ifnone
import inspect as inspect_builtin


# %% ../../nbs/config.ipynb 2
class DeepLearningConfig:
    """
    A configuration class for deep learning models.

    This class provides methods to access and manipulate configuration settings.

    Attributes:
        input_variables (list): List of input variables defined in the class constructor.

    Methods:
        keys(): Returns the list of input variables.
        __getitem__(key): Returns the value of the specified key.
        __contains__(key): Checks if the specified key is present in the input variables.
        from_ini_file(filepath, config_root=None): Creates an instance of the class from an INI file.
        __repr__(): Returns a string representation of the class.

    Example usage:
        config = DeepLearningConfig()
        config.from_ini_file('config.ini')
        print(config.keys())
        print(config['learning_rate'])
    """

    def keys(self):
        if not hasattr(self, "input_variables"):
            self.input_variables = inspect_builtin.signature(
                self.__init__
            ).parameters.keys()
        return self.input_variables

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return key in self.input_variables

    @classmethod
    def from_ini_file(cls, filepath, *, config_root=None):
        config = parse(filepath)
        config_root = ifnone(config_root, getattr(cls, "config_root"))
        if config_root is not None:
            for _root in config_root.split("."):
                config = config[_root]
        return cls(**config)

    def __repr__(self):
        return f"{self.__class__.__name__}:\n" + str({**self})


# %% ../../nbs/config.ipynb 7
class GenericConfig(DeepLearningConfig):
    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.input_variables = kwargs.keys()

    @classmethod
    def from_ini_file(cls, filepath, *, config_root):
        config = parse(filepath)
        for _root in config_root.split("."):
            config = config[_root]
        # convert string type to list type
        return cls(**config)
