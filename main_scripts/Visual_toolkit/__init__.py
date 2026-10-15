"""Step 3.1 core.

Importing this package returns toolkit.py, so ``mock.patch`` on
``Visual_toolkit.BLOCK_SIZE_PIXELS`` still reaches the functions. The other
files in this folder stay importable as submodules.
"""
import sys
from pathlib import Path

from . import toolkit as _toolkit

_toolkit.__path__ = [str(Path(__file__).resolve().parent)]
_toolkit.__package__ = __package__ or "Visual_toolkit"
sys.modules[__name__] = _toolkit
