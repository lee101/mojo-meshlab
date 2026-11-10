"""Before/after numbers for the parallel experiment, measured on this host.

Run through ``pixi run bench-par`` so it takes the machine-wide benchmark lock.

The experiment added a `max.algorithm.parallelize` path to the per-face kernels,
then removed it: it lost at every size real callers pass (the dispatcher costs
more per call than these kernels take to run). This script produces the numbers
quoted in README "Benchmarks":

* "after" - the current shared library through the public Python filters;
* "before" - the same public filters on the pre-experiment library, built here
  from git into a temporary file, so no checkout or worktree change is needed;
* the floor - one ctypes call into the current library that does almost no work,
  which bounds what any per-call dispatcher has to beat.

Both libraries are loaded in separate processes because they export the same
symbols; each process prints one JSON line that this script's parent merges.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import platform
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SIZES = [4_096, 16_384, 65_536, 262_144, 403_202]
REPEAT = 7


def grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.meshgrid(np.linspace(0, 1, size), np.linspace(0, 1, size))
    z = 0.025 * np.sin(13 * x) * np.cos(11 * y)
    vertices = np.ascontiguousarray(np.column_stack((x.ravel(), y.ravel(), z.ravel())))
    faces = np.empty((2 * (size - 1) ** 2, 3), dtype=np.int64)
    cursor = 0
    for row in range(size - 1):
        base = row * size
        for col in range(size - 1):
            a = base + col
            faces[cursor] = (a, a + 1, a + size)
            faces[cursor + 1] = (a + 1, a + size + 1, a + size)
            cursor += 2
    return vertices, faces


def mesh_of(faces: int) -> tuple[np.ndarray, np.ndarray]:
    side = math.isqrt(faces // 2 + 1) + 1
    vertices, grid_faces = grid(side)
    grid_faces = np.ascontiguousarray(grid_faces[: faces - faces % 2])
    return vertices[: int(grid_faces.max()) + 1], grid_faces


def time_best(function, repeat: int = REPEAT) -> float:
    function()  # warm-up
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best * 1000.0


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


FILTERS = {
    "aspect ratio": ("compute_scalar_by_aspect_ratio_per_face", {"metric": 1}),
    "geometric measures": ("get_geometric_measures", {}),
    "select long edges": ("compute_selection_by_edge_length", {"threshold": 0.0}),
    "remove null faces": ("meshing_remove_null_faces", {}),
}


def measure_with(library_path: str) -> dict:
    """Measure every filter at every size against one shared library."""
    import mojomeshlab._lib as bindings
    import mojomeshlab.mesh as mesh_module

    bindings._library = ctypes.CDLL(library_path)
    for name, (argtypes, restype) in bindings._SIGNATURES.items():
        function = getattr(bindings._library, name)
        function.argtypes = argtypes
        function.restype = restype

    results = {}
    for size in SIZES:
        vertices, faces = mesh_of(size)
        mesh_set = bindings_meshset(mesh_module, vertices, faces)
        for label, (name, kwargs) in FILTERS.items():
            run = getattr(mesh_set, name)
            results[f"{label}|{len(faces)}"] = time_best(lambda: run(**kwargs), 3)

    # The floor: a ctypes call that selects on a single face and returns.
    sel = np.empty(1, dtype=np.int64)
    vertices, faces = mesh_of(SIZES[0])
    floor = time_best(
        lambda: bindings._library.mml_select_long_edges(
            bindings.addr(vertices), bindings.addr(faces), 1, 0.0, bindings.addr(sel)
        ),
        repeat=200,
    )
    return {"filters": results, "floor_ms": floor}


def bindings_meshset(mesh_module, vertices, faces):
    mesh_set = mesh_module.MeshSet()
    mesh_set.add_mesh(mesh_module.Mesh(vertices, faces))
    return mesh_set


def child(library_path: str) -> None:
    print(json.dumps(measure_with(library_path)))


def parent() -> None:
    sys.path.insert(0, os.path.join(ROOT, "python"))
    import mojomeshlab._lib as bindings

    after_library = bindings.build()
    before_source = os.path.join(tempfile.mkdtemp(prefix="mml-before-"), "meshlab.mojo")
    with open(before_source, "w", encoding="utf8") as handle:
        handle.write(
            subprocess.run(
                ["git", "show", "HEAD:src/meshlab.mojo"],
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
    before_library = os.path.join(os.path.dirname(before_source), "libbefore.so")
    subprocess.run(
        ["mojo", "build", "--emit", "shared-lib", before_source, "-o", before_library],
        cwd=ROOT,
        check=True,
        capture_output=True,
    )

    results = {}
    for label, path in (("before", before_library), ("after", after_library)):
        output = subprocess.run(
            [sys.executable, os.path.abspath(__file__), "--child", path],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip().splitlines()[-1]
        results[label] = json.loads(output)

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print("before = HEAD kernel library (serial), after = current serial kernel library")
    print()
    print("| filter | faces | before | after | ratio |")
    print("|---|---:|---:|---:|---:|")
    for size in SIZES:
        for label in FILTERS:
            before = results["before"]["filters"][f"{label}|{size}"]
            after = results["after"]["filters"][f"{label}|{size}"]
            print(f"| {label} | {size:,} | {before:.2f} ms | {after:.2f} ms | {before / after:.2f}x |")
    print()
    print(f"| floor: one ctypes call, 1 face | | | {results['after']['floor_ms'] * 1000:.1f} us | |")
    print()
    print("The parallel path was measured separately: dispatching it from this")
    print("ctypes host cost milliseconds per call, far above the floor above, and")
    print("the kernels themselves take 0.01-74 ms, so it never paid off. See README.")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        child(sys.argv[2])
    else:
        parent()
