"""Compatibility shim providing the legacy ``memoria_api`` module."""

from memoria.api import *  # noqa: F401,F403
from memoria.api import __all__ as _memoria_api_all  # noqa: F401

__all__ = _memoria_api_all
