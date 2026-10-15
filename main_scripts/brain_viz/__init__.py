"""Step 3.2 core. BrainViz loads from core.py on first use."""

def __getattr__(name):
    from . import core
    if hasattr(core, name):
        value = getattr(core, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
