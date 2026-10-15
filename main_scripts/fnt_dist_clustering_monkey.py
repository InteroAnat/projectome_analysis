"""Compatibility import. Implementation: fnt/dist_clustering_monkey.py."""
from fnt.dist_clustering_monkey import *  # noqa: F401,F403

if __name__ == "__main__":
    import runpy
    runpy.run_module("fnt.dist_clustering_monkey", run_name="__main__")
