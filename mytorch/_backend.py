"""
    A simple partern to get GPU acceleration. 
    It imports cupy (same interface as numpy but runs on GPU) if cuda is available,
    imports numpy otherwise. 
"""
import os

_device = os.environ.get("MYTORCH_DEVICE", "cpu")

if _device == "cuda":
    import cupy as xp
else:
    import numpy as xp

def get_device():
    return _device