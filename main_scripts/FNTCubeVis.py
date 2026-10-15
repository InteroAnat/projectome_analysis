"""Compatibility import. Implementation: fnt/cube_vis.py."""
from fnt.cube_vis import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("fnt.cube_vis", run_name="__main__")
