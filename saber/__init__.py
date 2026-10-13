"""
SABER: A deep learning framework for bulk RNA-seq deconvolution
"""

__all__ = ["SABER"]

def __getattr__(name):
    if name == "SABER":
        from .models.saber import SABER
        return SABER
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")