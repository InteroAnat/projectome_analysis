"""Compatibility import. Implementation: Visual_toolkit/context_resources.py."""
import sys

from Visual_toolkit import context_resources as _impl

sys.modules[__name__] = _impl
