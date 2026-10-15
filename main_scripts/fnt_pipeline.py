"""Compatibility import. Implementation: fnt/pipeline.py."""
from fnt.pipeline import FNTNeuronPipeline, run_pipeline, run_pipeline_generic

if __name__ == "__main__":
    import runpy
    runpy.run_module("fnt.pipeline", run_name="__main__")
