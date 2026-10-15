"""Step 3.2 core.

Importing this package returns core.py. Mesh wrappers in this folder stay
importable as submodules.
"""
import sys
from pathlib import Path

from . import core as _core

_core.__path__ = [str(Path(__file__).resolve().parent)]
_core.__package__ = __package__ or "brain_viz"
sys.modules[__name__] = _core
