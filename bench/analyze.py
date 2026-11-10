"""ANALYZE-phase measurement harness for mojo-meshlab.

Reports, for every public MeshSet entry point at sizes real callers use:

* wall  - best-of-N wall time, what a caller sees
* cpu   - process CPU time of the same best window (the ~0.9 us ctypes call
          on this machine is not billed by the scheduler, so cpu is a proxy
          for kernel work)
* glue  - wall - cpu, the Python/numpy/ctypes part outside the kernel
* rsd   - spread of the repeat set, so a noisy row is visible in the report

and an `upstream` column from pymeshlab with the same mesh. Run it under the
bench lock only:  flock /tmp/mojo-bench.lock pixi run python -u bench/analyze.py
"""

from __future__ import annotations

import ctypes
import gc
import json
import math
import os
import platform
import statistics
import sys
import time

import numpy as np
import pymeshlab as reference

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "python"))

import mojomeshlab as mojo  # noqa: E402


CPUS = len(os.sched_getaffinity(0))


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


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


class Timer:
    """Times repeat best-of-N, keeping the whole repeat set for the spread."""

    def __init__(self, repeat: int = 5):
        self.repeat = repeat

    def __call__(self, function) -> dict:
        function()  # warm: first ctypes/binding touch, cache warm-up
        walls, cpus = [], []
        for _ in range(self.repeat):
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            function()
            cpu_end = time.process_time()
            wall_end = time.perf_counter()
            walls.append(wall_end - wall_start)
            cpus.append(cpu_end - cpu_start)
        best = min(range(len(walls)), key=lambda index: walls[index])
        return {
            "wall": walls[best],
            "cpu": cpus[best],
            "median": statistics.median(walls),
            "rsd": statistics.pstdev(walls) / statistics.fmean(walls),
            "n": self.repeat,
        }


_LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
CLOCK_THREAD_CPUTIME_ID = 3


class _Timespec(ctypes.Structure):
    _fields_ = [("seconds", ctypes.c_long), ("nanoseconds", ctypes.c_long)]


def thread_cpu() -> int:
    """CLOCK_THREAD_CPUTIME_ID in ns: the calling Python thread's CPU time.

    Unlike process CPU time it is not inflated by parallel workers, and unlike
    perf_counter it excludes time spent descheduled by the bench lock's other
    users, which is what lets kernel and glue time be separated on a busy box.
    """
    value = _Timespec()
    _LIBC.clock_gettime(CLOCK_THREAD_CPUTIME_ID, ctypes.byref(value))
    return value.seconds * 1_000_000_000 + value.nanoseconds


# Kernel symbols each public entry point calls, in call order. Used to clock
# the mml_* call itself instead of having to edit src/ or the Python wrapper.
KERNELS = {
    "mml_geometric_measures": ["mml_geometric_measures"],
    "mml_face_metrics": ["mml_face_metrics"],
    "mml_color_ramp": ["mml_color_ramp"],
    "mml_select_long_edges": ["mml_select_long_edges"],
    "mml_select_scalar": ["mml_select_scalar"],
    "mml_mark_null_faces": ["mml_mark_null_faces"],
    "mml_duplicate_vertex_remap": ["mml_duplicate_vertex_remap"],
    "mml_mark_duplicate_faces": ["mml_mark_duplicate_faces"],
    "mml_midpoint_subdivide": ["mml_midpoint_subdivide"],
    "mml_geodesic": ["mml_geodesic"],
    "mml_sample_barycentric": ["mml_sample_barycentric"],
    "mml_laplacian_step": ["mml_laplacian_step"],
}


class KernelClock:
    """Wraps the ctypes CDLL so every mml_* call is timed in thread-CPU ns."""

    def __init__(self):
        self.total = 0
        self.calls = 0

    def reset(self):
        self.total = 0
        self.calls = 0

    def install(self):
        library = mojo._lib.lib()
        for name in KERNELS:
            function = getattr(library, name)
            if getattr(function, "_clocked", False):
                continue
            self._wrap(function, name)

    def _wrap(self, function, name):
        original = function

        def timed(*args):
            start = thread_cpu()
            try:
                return original(*args)
            finally:
                self.total += thread_cpu() - start
                self.calls += 1

        timed._clocked = True
        setattr(mojo._lib.lib(), name, timed)


def make_mojo(vertices, faces, **kwargs) -> mojo.MeshSet:
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh(vertices, faces, **kwargs))
    return mesh_set


def make_reference(vertices, faces, **kwargs) -> reference.MeshSet:
    mesh_set = reference.MeshSet()
    mesh_set.add_mesh(reference.Mesh(vertices, faces.astype(np.int32), **kwargs))
    return mesh_set


def measure(name, builder, call, upstream_call, repeat, clock):
    """builder -> a fresh object per timed call (mutating filters need this)."""
    timer = Timer(repeat)

    def timed_ours():
        clock.reset()
        call(builder())

    ours = timer(timed_ours)
    ours["kernels"] = clock.calls
    ours["kernel"] = clock.total
    theirs = None
    if upstream_call is not None:
        try:
            theirs = timer(lambda: upstream_call())
        except Exception as error:  # noqa: BLE001 - record, never drop the row
            theirs = {"error": f"{type(error).__name__}: {error}"}
    return {"name": name, "ours": ours, "upstream": theirs}


def main() -> None:
    gc.collect()
    rows = []
    clock = KernelClock()
    mojo._lib.lib()  # load the shared library before wrapping its symbols
    clock.install()

    # ---- 403k triangles: the size bench/bench.py and the tests use ----------
    big_v, big_f = grid(450)
    small_v, small_f = grid(180)
    # Clean grids: duplicate / null / unreferenced filters are quadratic in the
    # port, so they are measured on the sizes a caller would plausibly run them.
    dup_f = np.concatenate((big_f[:2000], big_f[1000:3000]))
    dup_v = np.concatenate((small_v, small_v[:500]))
    unreferenced_v = np.concatenate((small_v, small_v[:1000] + 100.0))

    def big_mojo():
        return make_mojo(big_v, big_f)

    def big_ref():
        return make_reference(big_v, big_f)

    for label, fn, up, repeat in [
        ("meshing_surface_subdivision_midpoint it=1", lambda s: s.meshing_surface_subdivision_midpoint(iterations=1, threshold=0.0),
         lambda s: s.meshing_surface_subdivision_midpoint(iterations=1, threshold=reference.PureValue(0)), 5),
        ("compute_scalar_by_aspect_ratio_per_face metric=1", lambda s: s.compute_scalar_by_aspect_ratio_per_face(metric=1),
         lambda s: s.compute_scalar_by_aspect_ratio_per_face(metric=1), 5),
        ("compute_scalar_by_aspect_ratio_per_face metric=0", lambda s: s.compute_scalar_by_aspect_ratio_per_face(metric=0),
         None, 5),
        ("get_geometric_measures", lambda s: s.get_geometric_measures(),
         lambda s: s.get_geometric_measures(), 5),
        ("get_topological_measures", lambda s: s.get_topological_measures(),
         lambda s: s.get_topological_measures(), 5),
        ("compute_selection_by_edge_length", lambda s: s.compute_selection_by_edge_length(threshold=0.01),
         lambda s: s.compute_selection_by_edge_length(threshold=reference.PureValue(0.01)), 5),
    ]:
        rows.append(measure(
            f"{label} | 403k tri",
            big_mojo,
            lambda s: fn(s),
            (lambda: up(big_ref())) if up else None,
            repeat,
        ))

    # Mutating selection on a shared set is fine to re-time (idempotent), but
    # scalar selection/colour need fresh state, so they use builders too.
    scalar = np.ascontiguousarray(np.linspace(-1, 1, len(big_v)))

    def sel_mojo():
        return make_mojo(big_v, big_f, v_scalar_array=scalar)

    def sel_ref():
        return make_reference(big_v, big_f, v_scalar_array=scalar)

    rows.append(measure(
        "compute_selection_by_scalar_per_vertex | 403k tri",
        sel_mojo,
        lambda s: s.compute_selection_by_scalar_per_vertex(minq=-0.5, maxq=0.5),
        lambda: sel_ref().compute_selection_by_scalar_per_vertex(
            minq=reference.PureValue(-0.5), maxq=reference.PureValue(0.5)),
        5,
        clock,
    ))
    rows.append(measure(
        "compute_color_from_scalar_per_vertex | 403k tri",
        sel_mojo,
        lambda s: s.compute_color_from_scalar_per_vertex(minval=-1.0, maxval=1.0),
        lambda: sel_ref().compute_color_from_scalar_per_vertex(
            minval=reference.PureValue(-1.0), maxval=reference.PureValue(1.0)),
        5,
        clock,
    ))

    # ---- 1m vertices: the colour ramp case in bench/bench.py ----------------
    million_v = np.ascontiguousarray(np.column_stack(
        (np.linspace(0, 1, 1_000_000), np.zeros(1_000_000), np.zeros(1_000_000))))
    million_scalar = np.ascontiguousarray(np.linspace(-1, 1, 1_000_000))
    rows.append(measure(
        "compute_color_from_scalar_per_vertex | 1m vert",
        lambda: make_mojo(million_v, np.empty((0, 3), np.int64), v_scalar_array=million_scalar),
        lambda s: s.compute_color_from_scalar_per_vertex(minval=-1.0, maxval=1.0),
        lambda: make_reference(million_v, np.empty((0, 3), np.int32), v_scalar_array=million_scalar)
        .compute_color_from_scalar_per_vertex(minval=reference.PureValue(-1.0), maxval=reference.PureValue(1.0)),
        5,
        clock,
    ))
    rows.append(measure(
        "compute_selection_by_scalar_per_vertex | 1m vert",
        lambda: make_mojo(million_v, np.empty((0, 3), np.int64), v_scalar_array=million_scalar),
        lambda s: s.compute_selection_by_scalar_per_vertex(minq=-0.5, maxq=0.5),
        None,
        5,
        clock,
    ))

    # ---- clean filters at the sizes that are safe for their complexity ------
    def dupf_mojo():
        mesh_set = make_mojo(big_v, dup_f)
        return mesh_set

    def dupf_ref():
        mesh_set = make_reference(big_v, dup_f)
        return mesh_set

    for count in (10_000, 50_000):
        f = dup_f[:count]
        rows.append(measure(
            f"meshing_remove_duplicate_faces | {count} tri",
            lambda f=f: make_mojo(big_v, f),
            lambda s: s.meshing_remove_duplicate_faces(),
            lambda f=f: make_reference(big_v, f).meshing_remove_duplicate_faces(),
            3,
        clock,
        ))
    rows.append(measure(
        "meshing_remove_duplicate_vertices | 6.5k vert (128 dup)",
        lambda: make_mojo(dup_v, small_f),
        lambda s: s.meshing_remove_duplicate_vertices(),
        lambda: make_reference(dup_v, small_f).meshing_remove_duplicate_vertices(),
        3,
        clock,
    ))
    rows.append(measure(
        "meshing_remove_null_faces | 403k tri",
        big_mojo,
        lambda s: s.meshing_remove_null_faces(),
        lambda: big_ref().meshing_remove_null_faces(),
        5,
        clock,
    ))
    rows.append(measure(
        "meshing_remove_unreferenced_vertices | 6.5k vert",
        lambda: make_mojo(unreferenced_v, small_f),
        lambda s: s.meshing_remove_unreferenced_vertices(),
        lambda: make_reference(unreferenced_v, small_f).meshing_remove_unreferenced_vertices(),
        5,
        clock,
    ))
    rows.append(measure(
        "generate_sampling_montecarlo 1000 samples | 403k tri",
        big_mojo,
        lambda s: s.generate_sampling_montecarlo(samplenum=1000, seed=0),
        lambda: big_ref().generate_sampling_montecarlo(samplenum=1000),
        5,
        clock,
    ))
    rows.append(measure(
        "generate_sampling_montecarlo 1m samples | 403k tri",
        big_mojo,
        lambda s: s.generate_sampling_montecarlo(samplenum=1_000_000, seed=0),
        lambda: big_ref().generate_sampling_montecarlo(samplenum=1_000_000),
        3,
        clock,
    ))
    rows.append(measure(
        "apply_coord_laplacian_smoothing step=1 | 403k tri",
        big_mojo,
        lambda s: s.apply_coord_laplacian_smoothing(stepsmoothnum=1),
        lambda: big_ref().apply_coord_laplacian_smoothing(stepsmoothnum=reference.PureValue(1)),
        3,
        clock,
    ))
    rows.append(measure(
        "meshing_decimation_clustering 0.01 | 403k tri",
        big_mojo,
        lambda s: s.meshing_decimation_clustering(threshold=0.01),
        lambda: big_ref().meshing_decimation_clustering(threshold=reference.PureValue(0.01)),
        3,
        clock,
    ))
    rows.append(measure(
        "meshing_invert_face_orientation | 403k tri",
        big_mojo,
        lambda s: s.meshing_invert_face_orientation(),
        lambda: big_ref().meshing_invert_face_orientation(),
        5,
        clock,
    ))

    # ---- geodesics: the only whole-mesh non-linear kernel -------------------
    for size, label in ((48, "4.4k tri"), (95, "17.6k tri")):
        gv, gf = grid(size)
        rows.append(measure(
            f"compute_scalar_by_geodesic_distance_from_given_point | {label}",
            lambda gv=gv, gf=gf: make_mojo(gv, gf),
            lambda s: s.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
                startpoint=(0.5, 0.5, 0.0)),
            lambda gv=gv, gf=gf: make_reference(gv, gf)
            .compute_scalar_by_geodesic_distance_from_given_point_per_vertex(startpoint=(0.5, 0.5, 0.0)),
            3,
        clock,
        ))
        rows.append(measure(
            f"compute_scalar_by_geodesic_distance_from_border | {label}",
            lambda gv=gv, gf=gf: make_mojo(gv, gf),
            lambda s: s.compute_scalar_by_geodesic_distance_from_border_per_vertex(),
            None,
            3,
        clock,
        ))

    # ---- constructors and Mesh() glue --------------------------------------
    rows.append(measure(
        "create_sphere subdiv=3 | 1.3k tri",
        lambda: mojo.MeshSet(),
        lambda s: s.create_sphere(subdiv=3),
        lambda: reference.MeshSet().create_sphere(subdiv=reference.PureValue(3)),
        5,
        clock,
    ))
    rows.append(measure(
        "create_sphere subdiv=5 | 20.5k tri",
        lambda: mojo.MeshSet(),
        lambda s: s.create_sphere(subdiv=5),
        lambda: reference.MeshSet().create_sphere(subdiv=reference.PureValue(5)),
        3,
        clock,
    ))
    rows.append(measure(
        "Mesh(vertices, faces) | 403k tri",
        lambda: None,
        lambda: mojo.Mesh(big_v, big_f),
        lambda: reference.Mesh(big_v, big_f.astype(np.int32)),
        5,
        clock,
    ))

    # ---- pure-numpy baselines for the same arithmetic ----------------------
    faces_f = big_f
    def baseline_areas():
        a = big_v[faces_f[:, 0]]
        b = big_v[faces_f[:, 1]]
        c = big_v[faces_f[:, 2]]
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()

    def baseline_aspect():
        a = big_v[faces_f[:, 0]]
        b = big_v[faces_f[:, 1]]
        c = big_v[faces_f[:, 2]]
        la = np.linalg.norm(b - a, axis=1)
        lb = np.linalg.norm(c - a, axis=1)
        lc = np.linalg.norm(c - b, axis=1)
        s = (la + lb + lc) * 0.5
        area2 = s * (la + lb - s) * (la + lc - s) * (lb + lc - s)
        return 8.0 * area2 / (la * lb * lc * s)

    def baseline_color():
        x = million_scalar - (-1.0)
        step = 2.0 / 4.0
        return np.clip(np.minimum(x, step) / step, 0.0, 1.0)

    for label, fn in (("numpy baseline: face areas | 403k tri", baseline_areas),
                      ("numpy baseline: aspect metric=1 | 403k tri", baseline_aspect),
                      ("numpy baseline: color ramp | 1m vert", baseline_color)):
        rows.append({"name": label, "ours": Timer(5)(fn), "upstream": None, "baseline": True})

    out = {
        "machine": cpu_name(),
        "os": f"{platform.system()} {platform.release()}",
        "cpus_affinity": CPUS,
        "rows": rows,
    }
    with open(os.path.join(HERE, "analyze.json"), "w", encoding="utf8") as sink:
        json.dump(out, sink, indent=1)

    print(f"{'case':48s} {'wall':>9s} {'cpu':>9s} {'kern':>9s} {'glue':>8s} {'nk':>2s} "
          f"{'rsd':>5s} {'upstream':>10s} {'x':>6s}   (ms)")
    for row in rows:
        ours = row["ours"]
        up = row["upstream"] or {}
        up_ms = f"{up['wall'] * 1000:.3f}" if "wall" in up else "n/a"
        ratio = f"{up['wall'] / ours['wall']:.2f}" if "wall" in up else "-"
        glue = ours["wall"] - ours.get("kernel", 0.0)
        print(f"{row['name']:48s} {ours['wall'] * 1000:9.3f} {ours['cpu'] * 1000:9.3f} "
              f"{ours.get('kernel', 0.0) * 1000:9.3f} {glue * 1000:8.3f} "
              f"{ours.get('kernels', 0):2d} {ours['rsd']:5.3f} {up_ms:>10s} {ratio:>6s}")
    print(f"\nwrote {os.path.join(HERE, 'analyze.json')}")


if __name__ == "__main__":
    main()
