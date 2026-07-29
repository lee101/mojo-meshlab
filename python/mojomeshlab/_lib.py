"""ctypes bindings for the Mojo mesh kernels."""

from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIBRARY = os.path.join(ROOT, "dist", "libmojo-meshlab.so")
I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mml_geometric_measures": ([I] * 11, None),
    "mml_face_metrics": ([I, I, I, I, I, I], None),
    "mml_color_ramp": ([I, I, F, F, I], None),
    "mml_select_long_edges": ([I, I, I, F, I], I),
    "mml_select_scalar": ([I, I, F, F, I], I),
    "mml_mark_null_faces": ([I, I, I, I], I),
    "mml_duplicate_vertex_remap": ([I, I, I], I),
    "mml_mark_duplicate_faces": ([I, I, I], I),
    "mml_midpoint_subdivide": ([I] * 10, I),
    "mml_geodesic": ([I] * 10, None),
    "mml_sample_barycentric": ([I, I, I, I, I, I], None),
    "mml_laplacian_step": ([I, I, I, I, I, F, I], None),
}

_library: ctypes.CDLL | None = None


def build() -> str:
    sources = [os.path.join(ROOT, "src", "meshlab.mojo")]
    if not os.path.exists(LIBRARY) or os.path.getmtime(LIBRARY) < max(map(os.path.getmtime, sources)):
        subprocess.run(["bash", os.path.join(ROOT, "build", "build.sh")], check=True)
    return LIBRARY


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
    return _library


def f64(values, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(values, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.float64)


def i64(values, *, copy: bool = False) -> np.ndarray:
    if copy:
        return np.array(values, dtype=np.int64, order="C", copy=True)
    return np.ascontiguousarray(values, dtype=np.int64)


def addr(array: np.ndarray) -> int:
    if not isinstance(array, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if not array.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(array.ctypes.data)
    if array.size and address == 0:
        raise ValueError("non-empty FFI buffer has a null address")
    return address
