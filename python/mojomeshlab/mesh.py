"""pymeshlab-shaped mesh containers and filter dispatch."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable

import numpy as np

from ._lib import addr, f64, i64, lib


def _empty_faces() -> np.ndarray:
    return np.empty((0, 3), dtype=np.int64)


def _matrix(values: Any, *, name: str, integer: bool) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    if integer:
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("face_matrix must contain integers")
        info = np.iinfo(np.int64)
        if array.size and (array.min() < info.min or array.max() > info.max):
            raise OverflowError("face_matrix values do not fit in int64")
        return np.array(array, dtype=np.int64, order="C", copy=True)
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("vertex_matrix must contain numbers")
    result = np.array(array, dtype=np.float64, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError("vertex_matrix must contain only finite values")
    return result


@dataclass
class Mesh:
    _vertices: np.ndarray
    _faces: np.ndarray

    def __init__(
        self,
        vertex_matrix: Any = None,
        face_matrix: Any = None,
        *,
        v_scalar_array: Any = None,
        f_scalar_array: Any = None,
    ):
        vertices = np.empty((0, 3)) if vertex_matrix is None else vertex_matrix
        faces = _empty_faces() if face_matrix is None else face_matrix
        self._vertices = _matrix(vertices, name="vertex_matrix", integer=False)
        self._faces = _matrix(faces, name="face_matrix", integer=True)
        if self._faces.size and (self._faces.min() < 0 or self._faces.max() >= len(self._vertices)):
            raise ValueError("face index outside the vertex matrix")
        self._v_scalar = (
            np.zeros(len(self._vertices))
            if v_scalar_array is None
            else f64(v_scalar_array, copy=True).reshape(-1)
        )
        self._f_scalar = (
            np.zeros(len(self._faces))
            if f_scalar_array is None
            else f64(f_scalar_array, copy=True).reshape(-1)
        )
        if len(self._v_scalar) != len(self._vertices) or len(self._f_scalar) != len(self._faces):
            raise ValueError("scalar array length does not match mesh")
        if not np.isfinite(self._v_scalar).all() or not np.isfinite(self._f_scalar).all():
            raise ValueError("scalar arrays must contain only finite values")
        self._v_color = np.ones((len(self._vertices), 4))
        self._f_color = np.ones((len(self._faces), 4))
        self._v_selected = np.zeros(len(self._vertices), dtype=bool)
        self._f_selected = np.zeros(len(self._faces), dtype=bool)
        self._f_faux = np.zeros((len(self._faces), 3), dtype=bool)
        self._measure_keys = np.empty(0, dtype=np.int64)
        self._measure_states = np.empty(0, dtype=np.uint8)
        self._measure_scratch = np.empty((0, 5))

    def vertex_matrix(self) -> np.ndarray:
        return self._vertices.copy()

    def face_matrix(self) -> np.ndarray:
        return self._faces.copy()

    def vertex_scalar_array(self) -> np.ndarray:
        return self._v_scalar.copy()

    def face_scalar_array(self) -> np.ndarray:
        return self._f_scalar.copy()

    def vertex_color_matrix(self) -> np.ndarray:
        return self._v_color.copy()

    def face_color_matrix(self) -> np.ndarray:
        return self._f_color.copy()

    def vertex_selection_array(self) -> np.ndarray:
        return self._v_selected.copy()

    def face_selection_array(self) -> np.ndarray:
        return self._f_selected.copy()

    def vertex_number(self) -> int:
        return len(self._vertices)

    def face_number(self) -> int:
        return len(self._faces)

    def is_point_cloud(self) -> bool:
        return self.face_number() == 0


def _cube(size: float) -> Mesh:
    half = size / 2.0
    vertices = np.array(
        [
            [-half, -half, -half], [half, -half, -half],
            [-half, half, -half], [half, half, -half],
            [-half, -half, half], [half, -half, half],
            [-half, half, half], [half, half, half],
        ]
    )
    faces = np.array(
        [
            [2, 1, 0], [1, 2, 3], [4, 2, 0], [2, 4, 6],
            [1, 4, 0], [4, 1, 5], [6, 5, 7], [5, 6, 4],
            [3, 6, 7], [6, 3, 2], [5, 3, 7], [3, 5, 1],
        ]
    )
    mesh = Mesh(vertices, faces)
    mesh._f_faux[:, 0] = True
    return mesh


def _tetrahedron() -> Mesh:
    return Mesh(
        [[1, 1, 1], [-1, 1, -1], [-1, -1, 1], [1, -1, -1]],
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [3, 2, 1]],
    )


def _octahedron() -> Mesh:
    return Mesh(
        [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]],
        [[0, 1, 2], [0, 2, 4], [0, 4, 5], [0, 5, 1],
         [3, 1, 5], [3, 5, 4], [3, 4, 2], [3, 2, 1]],
    )


def _icosahedron() -> Mesh:
    golden = (math.sqrt(5.0) + 1.0) / 2.0
    vertices = [
        [0, golden, 1], [0, golden, -1], [0, -golden, 1], [0, -golden, -1],
        [golden, 1, 0], [golden, -1, 0], [-golden, 1, 0], [-golden, -1, 0],
        [1, 0, golden], [-1, 0, golden], [1, 0, -golden], [-1, 0, -golden],
    ]
    faces = [
        [1, 0, 4], [0, 1, 6], [2, 3, 5], [3, 2, 7], [4, 5, 10],
        [5, 4, 8], [6, 7, 9], [7, 6, 11], [8, 9, 2], [9, 8, 0],
        [10, 11, 1], [11, 10, 3], [0, 8, 4], [0, 6, 9], [1, 4, 10],
        [1, 11, 6], [2, 5, 8], [2, 9, 7], [3, 10, 5], [3, 7, 11],
    ]
    return Mesh(vertices, faces)


def _adjacency(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    count = len(mesh._vertices)
    sets = [set() for _ in range(count)]
    for a, b, c in mesh._faces:
        sets[a].update((int(b), int(c)))
        sets[b].update((int(a), int(c)))
        sets[c].update((int(a), int(b)))
    offsets = np.zeros(count + 1, dtype=np.int64)
    for index, neighbors in enumerate(sets):
        offsets[index + 1] = offsets[index] + len(neighbors)
    flat = np.fromiter((item for neighbors in sets for item in sorted(neighbors)), dtype=np.int64)
    return offsets, flat


def _vertex_faces(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    count = len(mesh._vertices)
    incident = [[] for _ in range(count)]
    for face, triangle in enumerate(mesh._faces):
        for vertex in triangle:
            incident[int(vertex)].append(face)
    offsets = np.zeros(count + 1, dtype=np.int64)
    for index, faces in enumerate(incident):
        offsets[index + 1] = offsets[index] + len(faces)
    flat = np.fromiter((face for faces in incident for face in faces), dtype=np.int64)
    return offsets, flat


def _laplacian_adjacency(mesh: Mesh, smooth_boundary: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges, counts = _edge_data(mesh)
    neighbors = [[] for _ in range(len(mesh._vertices))]
    extra_self = np.zeros(len(mesh._vertices), dtype=np.int64)
    boundary_vertices = (
        set(map(int, np.unique(edges[counts == 1])))
        if smooth_boundary and len(edges)
        else set()
    )
    for (a_value, b_value), count in zip(edges, counts):
        a, b = int(a_value), int(b_value)
        if a in boundary_vertices:
            if count == 1:
                neighbors[a].append(b)
                extra_self[a] = 1
        else:
            neighbors[a].extend([b] * int(count))
        if b in boundary_vertices:
            if count == 1:
                neighbors[b].append(a)
                extra_self[b] = 1
        else:
            neighbors[b].extend([a] * int(count))
    offsets = np.zeros(len(neighbors) + 1, dtype=np.int64)
    for index, row in enumerate(neighbors):
        offsets[index + 1] = offsets[index] + len(row)
    flat = np.fromiter((item for row in neighbors for item in row), dtype=np.int64)
    return offsets, flat, extra_self


def _edge_data(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    if not len(mesh._faces):
        return np.empty((0, 2), dtype=np.int64), np.empty(0, dtype=np.int64)
    faces = mesh._faces
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0, return_counts=True)


class MeshSet:
    def __init__(self):
        self._meshes: list[Mesh] = []
        self._names: list[str] = []
        self._current = -1

    def add_mesh(self, mesh: Mesh, mesh_name: str = "") -> int:
        if not isinstance(mesh, Mesh):
            raise TypeError("mesh must be a mojomeshlab.Mesh")
        self._meshes.append(mesh)
        self._names.append(mesh_name)
        self._current = len(self._meshes) - 1
        return self._current

    def current_mesh(self) -> Mesh:
        if self._current < 0:
            raise RuntimeError("MeshSet has no current mesh")
        return self._meshes[self._current]

    def number_meshes(self) -> int:
        return len(self._meshes)

    def set_current_mesh(self, mesh_id: int) -> None:
        index = int(mesh_id)
        if index < 0 or index >= len(self._meshes):
            raise IndexError("mesh_id outside the MeshSet")
        self._current = index

    def __getattr__(self, name: str) -> Callable[..., dict[str, Any]]:
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda **kwargs: self.apply_filter(name, **kwargs)

    def apply_filter(self, filter_name: str, **kwargs) -> dict[str, Any]:
        dispatch = {
            "create_cube": self._create_cube,
            "create_tetrahedron": self._create_tetrahedron,
            "create_octahedron": self._create_octahedron,
            "create_icosahedron": self._create_icosahedron,
            "create_sphere": self._create_sphere,
            "meshing_surface_subdivision_midpoint": self._midpoint,
            "meshing_remove_duplicate_vertices": self._remove_duplicate_vertices,
            "meshing_remove_duplicate_faces": self._remove_duplicate_faces,
            "meshing_remove_null_faces": self._remove_null_faces,
            "meshing_remove_unreferenced_vertices": self._remove_unreferenced,
            "get_geometric_measures": self._geometric_measures,
            "get_topological_measures": self._topological_measures,
            "compute_scalar_by_geodesic_distance_from_given_point_per_vertex": self._geodesic_point,
            "compute_scalar_by_geodesic_distance_from_border_per_vertex": self._geodesic_border,
            "compute_scalar_by_border_distance_per_vertex": self._geodesic_border,
            "compute_selection_by_edge_length": self._select_edge,
            "compute_selection_by_scalar_per_vertex": self._select_vertex_scalar,
            "compute_color_from_scalar_per_vertex": self._color_vertex_scalar,
            "compute_scalar_by_aspect_ratio_per_face": self._face_quality,
            "generate_sampling_montecarlo": self._sample_montecarlo,
            "apply_coord_laplacian_smoothing": self._laplacian,
            "meshing_decimation_clustering": self._cluster_decimation,
            "meshing_invert_face_orientation": self._invert_faces,
        }
        try:
            function = dispatch[filter_name]
        except KeyError as error:
            raise ValueError(f"unsupported filter: {filter_name}") from error
        return function(**kwargs)

    def _create_cube(self, size: float = 1.0, **_: Any) -> dict[str, Any]:
        self.add_mesh(_cube(float(size)), "Box/Cube")
        return {}

    def _create_tetrahedron(self, **_: Any) -> dict[str, Any]:
        self.add_mesh(_tetrahedron(), "Tetrahedron")
        return {}

    def _create_octahedron(self, **_: Any) -> dict[str, Any]:
        self.add_mesh(_octahedron(), "Octahedron")
        return {}

    def _create_icosahedron(self, **_: Any) -> dict[str, Any]:
        self.add_mesh(_icosahedron(), "Icosahedron")
        return {}

    def _create_sphere(self, radius: float = 1.0, subdiv: int = 3, **_: Any) -> dict[str, Any]:
        if int(subdiv) != subdiv or subdiv < 0:
            raise ValueError("subdiv must be a non-negative integer")
        self.add_mesh(_icosahedron(), "Sphere")
        mesh = self.current_mesh()
        mesh._vertices /= np.linalg.norm(mesh._vertices, axis=1)[:, None]
        for _ in range(int(subdiv)):
            self._midpoint(iterations=1, threshold=0.0)
            mesh._vertices /= np.linalg.norm(mesh._vertices, axis=1)[:, None]
        mesh._vertices *= float(radius)
        return {}

    def _midpoint(self, iterations: int = 3, threshold: float = 0.0, selected: bool = False, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        if int(iterations) != iterations or iterations < 0:
            raise ValueError("iterations must be a non-negative integer")
        if selected:
            raise NotImplementedError("selected-only midpoint subdivision is not covered")
        for _ in range(int(iterations)):
            if not len(mesh._faces):
                break
            if threshold and max(
                np.linalg.norm(mesh._vertices[mesh._faces[:, 0]] - mesh._vertices[mesh._faces[:, 1]], axis=1).max(),
                np.linalg.norm(mesh._vertices[mesh._faces[:, 1]] - mesh._vertices[mesh._faces[:, 2]], axis=1).max(),
                np.linalg.norm(mesh._vertices[mesh._faces[:, 2]] - mesh._vertices[mesh._faces[:, 0]], axis=1).max(),
            ) < threshold:
                break
            maximum_vertices = len(mesh._vertices) + 3 * len(mesh._faces)
            vertices = np.empty((maximum_vertices, 3))
            faces = np.empty((4 * len(mesh._faces), 3), dtype=np.int64)
            table_size = 1
            while table_size < 8 * len(mesh._faces):
                table_size *= 2
            table_a = np.empty(table_size, dtype=np.int64)
            table_b = np.empty(table_size, dtype=np.int64)
            table_value = np.empty(table_size, dtype=np.int64)
            count = lib().mml_midpoint_subdivide(
                addr(mesh._vertices), addr(mesh._faces), len(mesh._vertices), len(mesh._faces),
                addr(vertices), addr(faces), addr(table_a), addr(table_b), addr(table_value), table_size,
            )
            mesh._vertices = np.ascontiguousarray(vertices[:count])
            mesh._faces = faces
            mesh._v_scalar = np.zeros(count)
            mesh._f_scalar = np.zeros(len(faces))
            mesh._v_color = np.ones((count, 4))
            mesh._f_color = np.ones((len(faces), 4))
            mesh._v_selected = np.zeros(count, dtype=bool)
            mesh._f_selected = np.zeros(len(faces), dtype=bool)
            mesh._f_faux = np.zeros((len(faces), 3), dtype=bool)
        return {}

    def _remove_null_faces(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        keep = np.empty(len(mesh._faces), dtype=np.int64)
        removed = lib().mml_mark_null_faces(
            addr(mesh._vertices), addr(mesh._faces), len(mesh._faces), addr(keep)
        ) if len(mesh._faces) else 0
        mask = keep.astype(bool)
        mesh._faces = np.ascontiguousarray(mesh._faces[mask])
        mesh._f_scalar = np.ascontiguousarray(mesh._f_scalar[mask])
        mesh._f_color = np.ascontiguousarray(mesh._f_color[mask])
        mesh._f_selected = np.ascontiguousarray(mesh._f_selected[mask])
        mesh._f_faux = np.ascontiguousarray(mesh._f_faux[mask])
        return {"removed_faces": int(removed)}

    def _remove_duplicate_faces(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        keep = np.empty(len(mesh._faces), dtype=np.int64)
        removed = lib().mml_mark_duplicate_faces(addr(mesh._faces), len(mesh._faces), addr(keep)) if len(mesh._faces) else 0
        mask = keep.astype(bool)
        mesh._faces = np.ascontiguousarray(mesh._faces[mask])
        mesh._f_scalar = np.ascontiguousarray(mesh._f_scalar[mask])
        mesh._f_color = np.ascontiguousarray(mesh._f_color[mask])
        mesh._f_selected = np.ascontiguousarray(mesh._f_selected[mask])
        mesh._f_faux = np.ascontiguousarray(mesh._f_faux[mask])
        return {"removed_faces": int(removed)}

    def _remove_duplicate_vertices(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        remap = np.empty(len(mesh._vertices), dtype=np.int64)
        removed = lib().mml_duplicate_vertex_remap(addr(mesh._vertices), len(mesh._vertices), addr(remap)) if len(mesh._vertices) else 0
        if removed:
            first = np.unique(remap, return_index=True)[1]
            order = first[np.argsort(remap[first])]
            mesh._vertices = np.ascontiguousarray(mesh._vertices[order])
            mesh._v_scalar = np.ascontiguousarray(mesh._v_scalar[order])
            mesh._v_color = np.ascontiguousarray(mesh._v_color[order])
            mesh._v_selected = np.ascontiguousarray(mesh._v_selected[order])
            mesh._faces = np.ascontiguousarray(remap[mesh._faces])
            self._remove_null_faces()
        return {"removed_vertices": int(removed)}

    def _remove_unreferenced(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        referenced = np.zeros(len(mesh._vertices), dtype=bool)
        if mesh._faces.size:
            referenced[mesh._faces.ravel()] = True
        remap = np.cumsum(referenced, dtype=np.int64) - 1
        removed = int((~referenced).sum())
        mesh._vertices = np.ascontiguousarray(mesh._vertices[referenced])
        mesh._v_scalar = np.ascontiguousarray(mesh._v_scalar[referenced])
        mesh._v_color = np.ascontiguousarray(mesh._v_color[referenced])
        mesh._v_selected = np.ascontiguousarray(mesh._v_selected[referenced])
        if mesh._faces.size:
            mesh._faces = np.ascontiguousarray(remap[mesh._faces])
        return {"removed_vertices": removed}

    def _geometric_measures(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        result = np.zeros(13)
        if len(mesh._vertices) and len(mesh._faces):
            table_size = 1
            while table_size < 4 * len(mesh._faces):
                table_size *= 2
            if len(mesh._measure_keys) != table_size:
                mesh._measure_keys = np.empty(table_size, dtype=np.int64)
                mesh._measure_states = np.empty(table_size, dtype=np.uint8)
            worker_count = (
                min(16, (len(mesh._faces) + 32_767) // 32_768)
                if len(mesh._faces) >= 65_536
                else 1
            )
            if len(mesh._measure_scratch) != worker_count:
                mesh._measure_scratch = np.empty((worker_count, 5))
            lib().mml_geometric_measures(
                addr(mesh._vertices), addr(mesh._faces), addr(mesh._f_faux),
                len(mesh._vertices), len(mesh._faces), addr(result), addr(mesh._measure_keys),
                addr(mesh._measure_states), table_size, addr(mesh._measure_scratch),
                worker_count,
            )
        elif len(mesh._vertices):
            result[7:10] = mesh._vertices.mean(axis=0)
        unique_sum = result[1]
        unique_count = int(result[2])
        real_sum = unique_sum - result[10]
        real_count = unique_count - int(result[11])
        answer: dict[str, Any] = {
            "surface_area": result[0],
            "total_edge_inc_faux_length": unique_sum,
            "avg_edge_inc_faux_length": unique_sum / unique_count if unique_count else 0.0,
            "shell_barycenter": result[3:6].copy(),
            "barycenter": result[7:10].copy(),
            "total_edge_length": real_sum,
            "avg_edge_length": real_sum / real_count if real_count else 0.0,
        }
        if result[12] != 0.0:
            answer["mesh_volume"] = result[6]
        return answer

    def _topological_measures(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        edges, counts = _edge_data(mesh)
        referenced = np.zeros(len(mesh._vertices), dtype=bool)
        if mesh._faces.size:
            referenced[mesh._faces.ravel()] = True
        parent = np.arange(len(mesh._vertices))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = int(parent[index])
            return index

        for a, b in edges:
            ra, rb = find(int(a)), find(int(b))
            parent[ra] = rb
        components = len({find(int(v)) for v in np.flatnonzero(referenced)}) if referenced.any() else 0
        boundary = int(np.count_nonzero(counts == 1))
        nonmanifold = int(np.count_nonzero(counts > 2))
        manifold = nonmanifold == 0
        holes = boundary  # replaced below by boundary-loop count
        if boundary:
            boundary_edges = edges[counts == 1]
            boundary_vertices = np.unique(boundary_edges)
            boundary_parent = {int(v): int(v) for v in boundary_vertices}

            def bfind(v: int) -> int:
                while boundary_parent[v] != v:
                    v = boundary_parent[v]
                return v

            for a, b in boundary_edges:
                ra, rb = bfind(int(a)), bfind(int(b))
                boundary_parent[ra] = rb
            holes = len({bfind(int(v)) for v in boundary_vertices})
        elif manifold:
            holes = 0
        else:
            holes = -1
        genus = (
            int((2 * components - holes - (int(referenced.sum()) - len(edges) + len(mesh._faces))) // 2)
            if manifold else -1
        )
        return {
            "vertices_number": len(mesh._vertices),
            "edges_number": len(edges),
            "faces_number": len(mesh._faces),
            "unreferenced_vertices": int((~referenced).sum()),
            "boundary_edges": boundary,
            "connected_components_number": components,
            "is_mesh_two_manifold": manifold,
            "non_two_manifold_edges": nonmanifold,
            "incident_faces_on_non_two_manifold_edges": int(counts[counts > 2].sum()),
            "number_holes": holes,
            "genus": genus,
        }

    def _geodesic(self, seeds: np.ndarray) -> None:
        mesh = self.current_mesh()
        offsets, incidents = _vertex_faces(mesh)
        visited = np.empty(len(mesh._vertices), dtype=np.int64)
        sources = np.empty(len(mesh._vertices), dtype=np.int64)
        mesh._v_scalar = np.empty(len(mesh._vertices))
        seeds = i64(seeds)
        if not len(seeds):
            mesh._v_scalar.fill(np.inf)
            return
        lib().mml_geodesic(
            addr(mesh._vertices), addr(mesh._faces), addr(offsets), addr(incidents),
            len(mesh._vertices), addr(seeds), len(seeds), addr(mesh._v_scalar),
            addr(visited), addr(sources),
        )

    def _geodesic_point(self, startpoint=(0.0, 0.0, 0.0), **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        if not len(mesh._vertices):
            return {}
        point = np.asarray(startpoint, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("startpoint must be a finite 3-vector")
        seed = int(np.argmin(np.sum((mesh._vertices - point) ** 2, axis=1)))
        self._geodesic(np.array([seed], dtype=np.int64))
        return {}

    def _geodesic_border(self, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        edges, counts = _edge_data(mesh)
        seeds = np.unique(edges[counts == 1]) if len(edges) else np.empty(0, dtype=np.int64)
        self._geodesic(seeds)
        return {}

    def _select_edge(self, threshold: float = 0.0, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        selected = np.empty(len(mesh._faces), dtype=np.int64)
        count = lib().mml_select_long_edges(
            addr(mesh._vertices), addr(mesh._faces), len(mesh._faces), float(threshold), addr(selected)
        ) if len(mesh._faces) else 0
        mesh._f_selected = selected.astype(bool)
        return {"selected_faces": int(count)}

    def _select_vertex_scalar(self, minq: float = 0.0, maxq: float = 1.0, inclusive: bool = True, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        selected = np.empty(len(mesh._vertices), dtype=np.int64)
        count = lib().mml_select_scalar(addr(mesh._v_scalar), len(mesh._vertices), float(minq), float(maxq), addr(selected)) if len(mesh._vertices) else 0
        mesh._v_selected = selected.astype(bool)
        if len(mesh._faces):
            corner = mesh._v_selected[mesh._faces]
            mesh._f_selected = corner.all(axis=1) if inclusive else corner.any(axis=1)
        return {"selected_vertices": int(count)}

    def _color_vertex_scalar(self, minval: float | None = None, maxval: float | None = None, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        if not len(mesh._vertices):
            return {}
        minimum = float(mesh._v_scalar.min() if minval is None else minval)
        maximum = float(mesh._v_scalar.max() if maxval is None else maxval)
        mesh._v_color = np.empty((len(mesh._vertices), 4))
        lib().mml_color_ramp(addr(mesh._v_scalar), len(mesh._vertices), minimum, maximum, addr(mesh._v_color))
        return {}

    def _face_quality(self, metric: int = 0, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        areas = np.empty(len(mesh._faces))
        mesh._f_scalar = np.empty(len(mesh._faces))
        if len(mesh._faces):
            lib().mml_face_metrics(
                addr(mesh._vertices), addr(mesh._faces), len(mesh._faces),
                int(metric), addr(areas), addr(mesh._f_scalar),
            )
        return {}

    def _sample_montecarlo(self, samplenum: int = 1000, seed: int = 0, **_: Any) -> dict[str, Any]:
        source = self.current_mesh()
        count = int(samplenum)
        if count < 0 or not len(source._faces):
            raise ValueError("samplenum must be non-negative and the mesh must have faces")
        areas = np.empty(len(source._faces))
        quality = np.empty(len(source._faces))
        lib().mml_face_metrics(
            addr(source._vertices), addr(source._faces), len(source._faces),
            3, addr(areas), addr(quality),
        )
        total_area = float(areas.sum())
        if not math.isfinite(total_area) or total_area <= 0.0:
            raise ValueError("sampling requires a mesh with positive finite surface area")
        rng = np.random.default_rng(seed)
        face_indices = np.searchsorted(np.cumsum(areas), rng.random(count) * total_area).astype(np.int64)
        barycentric = np.ascontiguousarray(rng.random((count, 2)))
        samples = np.empty((count, 3))
        if count:
            lib().mml_sample_barycentric(
                addr(source._vertices), addr(source._faces), addr(face_indices),
                addr(barycentric), count, addr(samples),
            )
        self.add_mesh(Mesh(samples), "Montecarlo Samples")
        return {}

    def _laplacian(self, stepsmoothnum: int = 1, boundary: bool = True, cotangentweight: bool = False, **_: Any) -> dict[str, Any]:
        if cotangentweight:
            raise NotImplementedError("cotangent weights are not covered")
        if int(stepsmoothnum) != stepsmoothnum or stepsmoothnum < 0:
            raise ValueError("stepsmoothnum must be a non-negative integer")
        mesh = self.current_mesh()
        if not len(mesh._vertices):
            return {}
        offsets, neighbors, extra_self = _laplacian_adjacency(mesh, boundary)
        for _ in range(int(stepsmoothnum)):
            result = np.empty_like(mesh._vertices)
            lib().mml_laplacian_step(
                addr(mesh._vertices), addr(offsets), addr(neighbors), addr(extra_self),
                len(mesh._vertices), 1.0, addr(result),
            )
            mesh._vertices = result
        return {}

    def _cluster_decimation(self, threshold: float = 0.01, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        cell = float(threshold)
        if cell <= 0.0 or not len(mesh._vertices):
            return {}
        keys = np.floor((mesh._vertices - mesh._vertices.min(axis=0)) / cell).astype(np.int64)
        _, inverse = np.unique(keys, axis=0, return_inverse=True)
        count = int(inverse.max()) + 1
        vertices = np.zeros((count, 3))
        np.add.at(vertices, inverse, mesh._vertices)
        vertices /= np.bincount(inverse)[:, None]
        faces = inverse[mesh._faces]
        keep = (faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 2] != faces[:, 0])
        mesh._vertices = np.ascontiguousarray(vertices)
        mesh._faces = np.ascontiguousarray(faces[keep], dtype=np.int64)
        mesh._v_scalar = np.zeros(count)
        mesh._f_scalar = np.zeros(len(mesh._faces))
        mesh._v_color = np.ones((count, 4))
        mesh._f_color = np.ones((len(mesh._faces), 4))
        mesh._v_selected = np.zeros(count, dtype=bool)
        mesh._f_selected = np.zeros(len(mesh._faces), dtype=bool)
        mesh._f_faux = np.zeros((len(mesh._faces), 3), dtype=bool)
        self._remove_duplicate_faces()
        self._remove_unreferenced()
        return {}

    def _invert_faces(self, forceflip: bool = True, onlyselected: bool = False, **_: Any) -> dict[str, Any]:
        mesh = self.current_mesh()
        if forceflip:
            mask = mesh._f_selected if onlyselected else np.ones(len(mesh._faces), dtype=bool)
            mesh._faces[mask] = mesh._faces[mask][:, [0, 2, 1]]
        return {}
