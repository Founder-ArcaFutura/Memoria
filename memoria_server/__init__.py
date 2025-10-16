"""Namespace package for the Memoria server components."""

from .api import create_app  # re-export for convenience

__all__ = ["create_app"]
