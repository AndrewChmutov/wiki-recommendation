import inspect
from abc import ABC, ABCMeta, abstractmethod
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, ClassVar, Self, TypeVar, overload

import tomllib


@overload
def _normalize_key(d: dict) -> dict:
    ...


@overload
def _normalize_key(d: list) -> list:
    ...


@overload
def _normalize_key(d: Any) -> Any:
    ...


def _normalize_key(d: dict | list | Any) -> dict | list | Any:
    if isinstance(d, dict):
        return {k.replace("-", "_"): (v) for k, v in d.items()}
    elif isinstance(d, list):
        return [_normalize_key(item) for item in d]
    return d


def _normalize_name(name: str) -> str:
    return name.lower()


_config_path = Path("configs/rfc.toml")
if not _config_path.is_file():
    raise RuntimeError(f"{_config_path} is not a file")

_config = tomllib.loads((_config_path.read_text()))
_config = _normalize_key(_config)

R = TypeVar("R")


class StaticConfig(ABCMeta):
    def __new__(cls: type, *args, **kwargs) -> type:
        clss = super().__new__(cls, *args, **kwargs)

        original_init = clss.__init__
        signature = inspect.signature(original_init)

        def _wrapper(self, *args, **kwargs):  # noqa: ANN001, ANN202
            # From direct configuration
            new_kwargs = {
                param.name: (
                    self._configurable_or_none(param.name) or  # Other configurables     # noqa: E501
                    self.config().get(param.name) or           # Configurable parameters # noqa: E501
                    param.default                              # Default values          # noqa: E501
                )
                for param in signature.parameters.values()
                if param.name != "self"
            }

            # Passing other Configurables
            return original_init(self, *args, **(new_kwargs | kwargs))

        clss.__init__ = _wrapper
        return clss


class Configurable(ABC, metaclass=StaticConfig):
    _elements: ClassVar[dict[str, type[Self]]] = {}
    _config: ClassVar[dict[str, Any]] = _config

    def __init_subclass__(cls) -> None:
        cls._elements[cls.config_name()] = cls

    @classmethod
    @abstractmethod
    def config_name(cls) -> str:
        ...

    @classmethod
    def config(cls) -> dict[str, Any]:
        return cls._config.get(cls.config_name(), {})

    @classmethod
    def _configurable_or_none(cls, name: str) -> Self | None:
        if clss := cls._elements.get(name):
            return clss()


def from_config(func: Callable[..., R]) -> Callable[..., R]:
    # Check whether the command is configured
    if name := _normalize_name(func.__name__) not in _config:
        raise RuntimeError(f"Function {name} is not configured")

    signature = inspect.signature(func)

    # Check whether all arguments are keyword arguments
    for param in signature.parameters.values():
        if param.name != "self" and param is inspect.Parameter.empty:
            raise RuntimeError(
                f"Function {name} has non-keyword argument {param.name}"
            )

    new_kwargs = {
        param.name: (
            Configurable._configurable_or_none(param.name) or  # Configurable objects    # noqa: E501
            _config[func.__name__].get(param.name, None) or    # Configurable parameters # noqa: E501
            param.default                                      # Default values
        )
        for param in signature.parameters.values()
    }

    @wraps(func)
    def _wrapper(**kwargs) -> R:
        return func(**(new_kwargs | kwargs))

    return _wrapper


