"""Locked same-process benchmarks against pymeshlab."""

from __future__ import annotations

import math
import os
import platform
import sys
import time

import numpy as np
import pymeshlab as reference

sys.path.insert(
    0,
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"),
)

import mojomeshlab as mojo  # noqa: E402


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


def time_best(function, repeat: int = 3) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def mesh_sets(vertices, faces):
    ours = mojo.MeshSet()
    ours.add_mesh(mojo.Mesh(vertices, faces))
    theirs = reference.MeshSet()
    theirs.add_mesh(reference.Mesh(vertices, faces.astype(np.int32)))
    return ours, theirs


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf8") as source:
            for line in source:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def main() -> None:
    vertices, faces = grid(450)
    ours, theirs = mesh_sets(vertices, faces)
    cases = [
        (
            "face aspect ratio, 403k triangles",
            lambda: ours.compute_scalar_by_aspect_ratio_per_face(metric=1),
            lambda: theirs.compute_scalar_by_aspect_ratio_per_face(metric=1),
            5,
        ),
        (
            "geometric measures, 403k triangles",
            ours.get_geometric_measures,
            theirs.get_geometric_measures,
            3,
        ),
    ]

    color_vertices = np.ascontiguousarray(np.column_stack(
        (np.linspace(0, 1, 1_000_000), np.zeros(1_000_000), np.zeros(1_000_000))
    ))
    scalar = np.ascontiguousarray(np.linspace(-1, 1, 1_000_000))
    color_ours = mojo.MeshSet()
    color_ours.add_mesh(mojo.Mesh(color_vertices, v_scalar_array=scalar))
    color_theirs = reference.MeshSet()
    color_theirs.add_mesh(reference.Mesh(color_vertices, v_scalar_array=scalar))
    cases.append(
        (
            "scalar color ramp, 1m vertices",
            lambda: color_ours.compute_color_from_scalar_per_vertex(minval=-1.0, maxval=1.0),
            lambda: color_theirs.compute_color_from_scalar_per_vertex(minval=-1.0, maxval=1.0),
            5,
        )
    )

    midpoint_vertices, midpoint_faces = grid(180)

    def midpoint_ours():
        mesh_set = mojo.MeshSet()
        mesh_set.add_mesh(mojo.Mesh(midpoint_vertices, midpoint_faces))
        mesh_set.meshing_surface_subdivision_midpoint(iterations=1, threshold=0)

    def midpoint_theirs():
        mesh_set = reference.MeshSet()
        mesh_set.add_mesh(reference.Mesh(midpoint_vertices, midpoint_faces.astype(np.int32)))
        mesh_set.meshing_surface_subdivision_midpoint(
            iterations=1, threshold=reference.PureValue(0)
        )

    cases.append(("midpoint subdivision, 64k triangles", midpoint_ours, midpoint_theirs, 3))

    print(f"Machine: {cpu_name()}, {platform.system()} {platform.release()}")
    print()
    print("| case | mojo-meshlab | pymeshlab | pymeshlab / Mojo | result |")
    print("|---|---:|---:|---:|---|")
    for name, ours_fn, theirs_fn, repeat in cases:
        ours_fn()
        theirs_fn()
        ours_time = time_best(ours_fn, repeat)
        theirs_time = time_best(theirs_fn, repeat)
        ratio = theirs_time / ours_time
        result = "faster" if ratio > 1 else "slower"
        print(
            f"| {name} | {ours_time * 1000:.2f} ms | {theirs_time * 1000:.2f} ms "
            f"| {ratio:.2f}x | Mojo {result} |"
        )


if __name__ == "__main__":
    main()
