import sys
import catalogue
import confection
from confection import Config, ConfigValidationError
from typing import TypeVar, Callable

_DIn = TypeVar("_DIn")

# Use typing_extensions for Python versions < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Protocol, Literal
else:
    from typing import Protocol, Literal  # noqa: F401


class Decorator(Protocol):
    """Protocol to mark a function as returning its child with identical signature."""

    def __call__(self, name: str) -> Callable[[_DIn], _DIn]: ...


class registry(confection.registry):
    @classmethod
    def create(cls, registry_name: str, entry_points: bool = False) -> None:
        """Create a new custom registry."""
        if hasattr(cls, registry_name):
            # raise ValueError(f"Registry '{registry_name}' already exists")
            return
        reg: Decorator = catalogue.create(
            "torch_snippets", registry_name, entry_points=entry_points
        )
        setattr(cls, registry_name, reg)


__all__ = ["Config", "registry", "ConfigValidationError"]
