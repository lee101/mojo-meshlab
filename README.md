# mojo-meshlab

`mojo-meshlab` ports a focused, compute-heavy part of the MeshLab filter layer to
Mojo and exposes it through a small pymeshlab-compatible Python API. It is a
standalone project: MeshLab and VCGLib are not linked at build or run time.

This is an independent, from-scratch implementation of a small subset of
[MeshLab](https://github.com/cnr-isti-vclab/meshlab) behavior. MeshLab is
GPL-3.0; this project does not copy or compile MeshLab/VCGLib code and does not
link to either project. pymeshlab is used only as a test and benchmark oracle.
Source comments identify the upstream algorithm whose observable behavior is
being matched. This repository is MIT licensed.

## Coverage

The current release covers:

- `filter_clean`: exact-position duplicate vertices, duplicate faces regardless
  of winding, exact-zero-area faces, and unreferenced vertices;
- `filter_meshing`: uniform midpoint subdivision, clustering decimation, and
  face-orientation inversion;
- `filter_measure`: surface area, edge lengths, shell/vertex barycenters,
  watertight volume, and basic edge-based topology counts;
- `filter_colorproc` and `filter_quality`: all four non-texture triangle aspect
  metrics and the VCGLib red-yellow-green-cyan-blue scalar ramp;
- `filter_select`: vertex scalar ranges and faces with an edge at or beyond a
  length threshold;
- `filter_sampling`: area-weighted Monte Carlo surface sampling;
- `filter_geodesic`: VCGLib's unfolded-triangle geodesic update from the nearest
  vertex to a point and from mesh borders;
- `filter_create`: cube, tetrahedron, octahedron, icosahedron, and recursively
  subdivided sphere; and
- classic Laplacian coordinate smoothing, including VCGLib's face-incidence
  weighting and boundary rule.

This is not a general MeshLab replacement. It has no GUI, file import/export,
rendering, plugin system, point-cloud processing beyond generated samples, or
arbitrary pymeshlab filter compatibility. Not covered are QEM edge-collapse
decimation, adaptive midpoint splits, isotropic remeshing,
butterfly/Loop/Catmull-Clark subdivision, hole closing, ball pivoting,
non-manifold vertex splitting, heat geodesics, Poisson sampling, Voronoi
processing, expression parsing from `filter_func`, texture/wedge attributes,
and the curvature-driven `filter_trioptimize` operations. The reported manifold
status detects non-two-manifold edges, not non-manifold vertices. Hole and genus
results are intended for ordinary manifold triangle surfaces.

The clustering decimator and high-level topology assembly use NumPy; the
geometric, quality, color, subdivision, cleanup marking, geodesic, sampling
interpolation, and smoothing loops execute in Mojo.

## Install

```bash
pixi install
pixi run build
pixi run test
```

The environment includes pymeshlab 2023.12.post2 solely as the parity oracle
and benchmark reference. The runtime package itself does not call pymeshlab.

## Usage

```python
import mojomeshlab as ml

meshes = ml.MeshSet()
meshes.create_cube(size=2.0)
meshes.meshing_surface_subdivision_midpoint(iterations=1, threshold=0.0)

mesh = meshes.current_mesh()
print(mesh.vertex_number(), mesh.face_number())  # 26 48
print(meshes.get_geometric_measures()["surface_area"])  # 24.0
```

`Mesh(vertex_matrix, face_matrix, v_scalar_array=..., f_scalar_array=...)`,
`MeshSet.add_mesh`, `MeshSet.current_mesh`, direct filter methods, and
`MeshSet.apply_filter("filter_name", ...)` follow pymeshlab's common calling
style for the covered filters.

## Correctness

The test suite makes numerical or behavioral comparisons with the real
pymeshlab binding to the same upstream C++ for primitives, subdivision,
cleanup, measures, topology, four quality metrics, selection, color mapping,
geodesics, and smoothing. Sampling and clustering use deterministic invariant
tests because their point/order choices are not stable reference outputs.
Degenerate coverage includes an empty mesh, a single triangle, duplicate
vertices, duplicate/reversed faces, exact-zero-area faces, a non-manifold edge,
and an unreferenced vertex.

```text
40 passed
```

## How it works

Python owns C-contiguous `float64` vertex and attribute arrays and `int64`
triangle index arrays. The shared library receives only integer addresses and
sizes over a C ABI; Mojo reconstructs `UnsafePointer` views and never owns or
frees Python memory. Topology is represented by flat edge hashes or CSR offset
and index arrays rather than half-edge pointer graphs. Midpoint subdivision
uses an open-addressed edge table supplied by Python, so each shared edge gets
one new vertex without per-edge allocation. Geometric measures likewise use a
reusable open-addressed edge table instead of sorting and uniquing three
temporary edge arrays. Their vertex barycenter loop uses native-width SIMD with
a scalar tail, and their independent face reduction switches to parallel CPU
workers at 65,536 faces.

The build deliberately compiles one Mojo translation unit:

```text
src/meshlab.mojo -> dist/libmojo-meshlab.so -> ctypes -> mojomeshlab
```

## Benchmarks

Run benchmarks only through `pixi run bench`; the task takes a machine-wide
file lock. These are best-of-three or best-of-five wall times measured on an
Intel Xeon E5-2697 v4 at 2.30 GHz, Linux 6.8.0-136-generic. They are the actual
output from this repository on 2026-07-29.

| case | mojo-meshlab | pymeshlab | pymeshlab / Mojo | result |
|---|---:|---:|---:|---|
| face aspect ratio, 403k triangles | 12.62 ms | 26.11 ms | 2.07x | Mojo faster |
| geometric measures, 403k triangles | 157.38 ms | 1008.35 ms | 6.41x | Mojo faster |
| scalar color ramp, 1m vertices | 10.30 ms | 80.56 ms | 7.82x | Mojo faster |
| midpoint subdivision, 64k triangles | 40.02 ms | 206.45 ms | 5.16x | Mojo faster |

No GPU path is included. The geometric face pass performs fewer than two
floating-point operations per byte of indexed face and vertex data, while the
edge hash pass has still lower arithmetic intensity. Host-device transfer and
launch overhead therefore cannot be justified for these kernels.
