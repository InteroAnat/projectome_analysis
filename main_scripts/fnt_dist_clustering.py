"""Compatibility import. Implementation: fnt/dist_clustering.py."""
import sys

from fnt import dist_clustering as _impl

if __name__ == "__main__":
    import runpy
    runpy.run_module("fnt.dist_clustering", run_name="__main__")
else:
    # Keep private helpers such as _align_annotations on the old import name.
    sys.modules[__name__] = _impl
