"""Compatibility import. Implementation: Visual_toolkit/image_geometry.py."""
import importlib.util
from pathlib import Path

_path = Path(__file__).resolve().parent / "Visual_toolkit" / "image_geometry.py"
_spec = importlib.util.spec_from_file_location("_fmost_image_geometry_impl", _path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
globals().update({name: getattr(_mod, name) for name in dir(_mod) if not name.startswith("_")})
