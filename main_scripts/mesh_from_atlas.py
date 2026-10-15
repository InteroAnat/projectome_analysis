"""Compatibility import. Implementation: brain_viz/mesh_from_atlas.py."""
from brain_viz.mesh_from_atlas import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("brain_viz.mesh_from_atlas", run_name="__main__")
