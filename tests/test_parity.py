from __future__ import annotations

import numpy as np
import pymeshlab as reference
import pytest

import mojomeshlab as mojo


def sets(vertices, faces):
    ours = mojo.MeshSet()
    ours.add_mesh(mojo.Mesh(vertices, faces))
    theirs = reference.MeshSet()
    theirs.add_mesh(reference.Mesh(np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int32)))
    return ours, theirs


def sorted_rows(values):
    values = np.asarray(values)
    order = np.lexsort(tuple(values[:, index] for index in reversed(range(values.shape[1]))))
    return values[order]


@pytest.mark.parametrize(
    "name,vertices,faces",
    [
        ("create_cube", 8, 12),
        ("create_tetrahedron", 4, 4),
        ("create_octahedron", 6, 8),
        ("create_icosahedron", 12, 20),
    ],
)
def test_primitive_parity(name, vertices, faces):
    ours, theirs = mojo.MeshSet(), reference.MeshSet()
    getattr(ours, name)()
    getattr(theirs, name)()
    assert ours.current_mesh().vertex_number() == vertices
    assert ours.current_mesh().face_number() == faces
    np.testing.assert_allclose(
        ours.current_mesh().vertex_matrix(), theirs.current_mesh().vertex_matrix(), atol=1e-15
    )
    np.testing.assert_array_equal(
        ours.current_mesh().face_matrix(), theirs.current_mesh().face_matrix()
    )


def test_cube_geometric_measures_parity():
    ours, theirs = mojo.MeshSet(), reference.MeshSet()
    ours.create_cube(size=2.0)
    theirs.create_cube(size=2.0)
    a, b = ours.get_geometric_measures(), theirs.get_geometric_measures()
    for key in (
        "surface_area", "mesh_volume", "total_edge_length", "avg_edge_length",
        "total_edge_inc_faux_length", "avg_edge_inc_faux_length",
    ):
        assert a[key] == pytest.approx(b[key], rel=1e-12, abs=1e-12)
    np.testing.assert_allclose(a["barycenter"], b["barycenter"], atol=1e-14)
    np.testing.assert_allclose(a["shell_barycenter"], b["shell_barycenter"], atol=1e-14)


def test_triangle_geometric_measures_parity():
    vertices = [[0, 0, 0], [2, 0, 0], [0, 3, 0]]
    faces = [[0, 1, 2]]
    ours, theirs = sets(vertices, faces)
    a, b = ours.get_geometric_measures(), theirs.get_geometric_measures()
    assert a["surface_area"] == pytest.approx(b["surface_area"])
    assert a["total_edge_length"] == pytest.approx(b["total_edge_length"])
    np.testing.assert_allclose(a["shell_barycenter"], b["shell_barycenter"])
    assert "mesh_volume" not in a


@pytest.mark.parametrize("face_count", [65_535, 65_536])
def test_geometric_measures_parallel_threshold_and_simd_tail(face_count):
    vertices = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [9.0, 8.0, 7.0]]
    )
    faces = np.tile(np.array([[0, 1, 2]], dtype=np.int64), (face_count, 1))
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh(vertices, faces))
    measures = mesh_set.get_geometric_measures()
    assert measures["surface_area"] == pytest.approx(3.0 * face_count)
    assert measures["total_edge_length"] == pytest.approx(2.0 + 3.0 + np.sqrt(13.0))
    np.testing.assert_allclose(measures["barycenter"], vertices.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(measures["shell_barycenter"], [2 / 3, 1, 0], atol=1e-14)
    assert "mesh_volume" not in measures


def test_cube_topology_parity():
    ours, theirs = mojo.MeshSet(), reference.MeshSet()
    ours.create_cube()
    theirs.create_cube()
    a, b = ours.get_topological_measures(), theirs.get_topological_measures()
    for key in (
        "vertices_number", "edges_number", "faces_number", "boundary_edges",
        "connected_components_number", "is_mesh_two_manifold", "number_holes", "genus",
    ):
        assert a[key] == b[key]


def test_open_triangle_topology_parity():
    ours, theirs = sets([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    a, b = ours.get_topological_measures(), theirs.get_topological_measures()
    for key in ("edges_number", "boundary_edges", "connected_components_number", "number_holes", "genus"):
        assert a[key] == b[key]


@pytest.mark.parametrize("shape", ["triangle", "square"])
def test_midpoint_subdivision_parity(shape):
    if shape == "triangle":
        vertices, faces = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]]), np.array([[0, 1, 2]])
    else:
        vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
        faces = np.array([[0, 1, 2], [1, 3, 2]])
    ours, theirs = sets(vertices, faces)
    ours.meshing_surface_subdivision_midpoint(iterations=1, threshold=0.0)
    theirs.meshing_surface_subdivision_midpoint(iterations=1, threshold=reference.PureValue(0))
    np.testing.assert_allclose(
        sorted_rows(ours.current_mesh().vertex_matrix()),
        sorted_rows(theirs.current_mesh().vertex_matrix()),
    )
    ours_faces = np.sort(ours.current_mesh().face_matrix(), axis=1)
    their_faces = np.sort(theirs.current_mesh().face_matrix(), axis=1)
    np.testing.assert_array_equal(sorted_rows(ours_faces), sorted_rows(their_faces))


def test_sphere_parity():
    ours, theirs = mojo.MeshSet(), reference.MeshSet()
    ours.create_sphere(radius=2.5, subdiv=2)
    theirs.create_sphere(radius=2.5, subdiv=2)
    assert ours.current_mesh().vertex_number() == theirs.current_mesh().vertex_number() == 162
    assert ours.current_mesh().face_number() == theirs.current_mesh().face_number() == 320
    np.testing.assert_allclose(
        np.linalg.norm(ours.current_mesh().vertex_matrix(), axis=1), 2.5, atol=1e-14
    )
    assert ours.get_geometric_measures()["surface_area"] == pytest.approx(
        theirs.get_geometric_measures()["surface_area"], rel=1e-8
    )


def test_remove_duplicate_vertices_parity():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 0]])
    faces = np.array([[0, 1, 2], [3, 1, 2]])
    ours, theirs = sets(vertices, faces)
    ours.meshing_remove_duplicate_vertices()
    theirs.meshing_remove_duplicate_vertices()
    assert ours.current_mesh().vertex_number() == theirs.current_mesh().vertex_number() == 3
    assert ours.current_mesh().face_number() == theirs.current_mesh().face_number() == 2
    np.testing.assert_allclose(
        sorted_rows(ours.current_mesh().vertex_matrix()),
        sorted_rows(theirs.current_mesh().vertex_matrix()),
    )


def test_remove_duplicate_vertices_removes_degenerate_face():
    vertices = [[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [0, 2, 3]]
    ours, theirs = sets(vertices, faces)
    ours.meshing_remove_duplicate_vertices()
    theirs.meshing_remove_duplicate_vertices()
    assert ours.current_mesh().face_number() == theirs.current_mesh().face_number() == 1


def test_remove_duplicate_faces_parity():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [2, 1, 0], [0, 1, 2]]
    ours, theirs = sets(vertices, faces)
    ours.meshing_remove_duplicate_faces()
    theirs.meshing_remove_duplicate_faces()
    assert ours.current_mesh().face_number() == theirs.current_mesh().face_number() == 1


def test_remove_null_faces_parity():
    vertices = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]]
    faces = [[0, 1, 2], [0, 1, 3]]
    ours, theirs = sets(vertices, faces)
    ours.meshing_remove_null_faces()
    theirs.meshing_remove_null_faces()
    np.testing.assert_array_equal(ours.current_mesh().face_matrix(), theirs.current_mesh().face_matrix())


def test_remove_unreferenced_vertices_parity():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [9, 9, 9]]
    faces = [[0, 1, 2]]
    ours, theirs = sets(vertices, faces)
    ours.meshing_remove_unreferenced_vertices()
    theirs.meshing_remove_unreferenced_vertices()
    np.testing.assert_allclose(ours.current_mesh().vertex_matrix(), theirs.current_mesh().vertex_matrix())


@pytest.mark.parametrize("metric", [0, 1, 2, 3])
def test_face_quality_parity(metric):
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [3, 0, 0]]
    faces = [[0, 1, 2], [0, 1, 3]]
    ours, theirs = sets(vertices, faces)
    ours.compute_scalar_by_aspect_ratio_per_face(metric=metric)
    theirs.compute_scalar_by_aspect_ratio_per_face(metric=metric)
    np.testing.assert_allclose(
        ours.current_mesh().face_scalar_array(),
        theirs.current_mesh().face_scalar_array(),
        rtol=1e-12,
        atol=1e-14,
    )


def test_selection_by_edge_length_parity():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [4, 0, 0]]
    faces = [[0, 1, 2], [1, 3, 2]]
    ours, theirs = sets(vertices, faces)
    ours.compute_selection_by_edge_length(threshold=2.0)
    theirs.compute_selection_by_edge_length(threshold=2.0)
    np.testing.assert_array_equal(
        ours.current_mesh().face_selection_array(),
        theirs.current_mesh().face_selection_array(),
    )


def test_selection_by_vertex_scalar_parity():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    scalar = np.array([0., 1., 2., 3.])
    ours = mojo.MeshSet()
    ours.add_mesh(mojo.Mesh(vertices, faces, v_scalar_array=scalar))
    theirs = reference.MeshSet()
    theirs.add_mesh(reference.Mesh(vertices, faces, v_scalar_array=scalar))
    ours.compute_selection_by_scalar_per_vertex(minq=1.0, maxq=2.0, inclusive=True)
    theirs.compute_selection_by_scalar_per_vertex(minq=1.0, maxq=2.0, inclusive=True)
    np.testing.assert_array_equal(
        ours.current_mesh().vertex_selection_array(),
        theirs.current_mesh().vertex_selection_array(),
    )
    np.testing.assert_array_equal(
        ours.current_mesh().face_selection_array(),
        theirs.current_mesh().face_selection_array(),
    )


def test_color_ramp_parity():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    scalar = np.array([0., 1., 2., 3.])
    ours = mojo.MeshSet()
    ours.add_mesh(mojo.Mesh(vertices, faces, v_scalar_array=scalar))
    theirs = reference.MeshSet()
    theirs.add_mesh(reference.Mesh(vertices, faces, v_scalar_array=scalar))
    ours.compute_color_from_scalar_per_vertex(minval=0.0, maxval=3.0)
    theirs.compute_color_from_scalar_per_vertex(minval=0.0, maxval=3.0, colormap=0)
    np.testing.assert_allclose(
        ours.current_mesh().vertex_color_matrix()[:, :3],
        theirs.current_mesh().vertex_color_matrix()[:, :3],
        atol=1 / 255,
    )


def test_point_geodesic_parity_across_triangle_strip():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]])
    faces = np.array([[0, 1, 2], [1, 3, 2]])
    ours, theirs = sets(vertices, faces)
    ours.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(startpoint=[0, 0, 0])
    theirs.compute_scalar_by_geodesic_distance_from_given_point_per_vertex(
        startpoint=[0, 0, 0], maxdistance=reference.PureValue(100)
    )
    np.testing.assert_allclose(
        ours.current_mesh().vertex_scalar_array(),
        theirs.current_mesh().vertex_scalar_array(),
        atol=1e-12,
    )


def test_border_geodesic_parity():
    vertices = np.array(
        [[0., 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0], [1, 1, 0]]
    )
    faces = np.array([[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]])
    ours, theirs = sets(vertices, faces)
    ours.compute_scalar_by_border_distance_per_vertex()
    theirs.compute_scalar_by_border_distance_per_vertex()
    np.testing.assert_allclose(
        ours.current_mesh().vertex_scalar_array(),
        theirs.current_mesh().vertex_scalar_array(),
        atol=1e-12,
    )


def test_laplacian_smoothing_parity():
    vertices = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, .4], [.5, .5, 1.]])
    faces = np.array([[0, 1, 4], [1, 3, 4], [3, 2, 4], [2, 0, 4]])
    ours, theirs = sets(vertices, faces)
    ours.apply_coord_laplacian_smoothing(stepsmoothnum=1, boundary=True, cotangentweight=False)
    theirs.apply_coord_laplacian_smoothing(stepsmoothnum=1, boundary=True, cotangentweight=False)
    np.testing.assert_allclose(
        ours.current_mesh().vertex_matrix(), theirs.current_mesh().vertex_matrix(), atol=1e-12
    )


def test_sampling_surface_and_count_invariants():
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]]))
    mesh_set.generate_sampling_montecarlo(samplenum=500, seed=7)
    points = mesh_set.current_mesh().vertex_matrix()
    assert points.shape == (500, 3)
    assert np.all(points[:, :2] >= 0)
    assert np.all(points[:, 0] + points[:, 1] <= 1 + 1e-14)
    np.testing.assert_allclose(points[:, 2], 0)


def test_invert_orientation_reverses_signed_volume():
    mesh_set = mojo.MeshSet()
    mesh_set.create_tetrahedron()
    vertices = mesh_set.current_mesh().vertex_matrix()
    before = mesh_set.current_mesh().face_matrix()

    def signed_volume(faces):
        a, b, c = (vertices[faces[:, i]] for i in range(3))
        return np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6

    first = signed_volume(before)
    mesh_set.meshing_invert_face_orientation()
    second = signed_volume(mesh_set.current_mesh().face_matrix())
    assert second == pytest.approx(-first)


def test_clustering_decimation_reduces_dense_grid():
    x, y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
    vertices = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    faces = []
    for row in range(19):
        for col in range(19):
            a = row * 20 + col
            faces.extend(([a, a + 1, a + 20], [a + 1, a + 21, a + 20]))
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh(vertices, faces))
    mesh_set.meshing_decimation_clustering(threshold=0.15)
    assert mesh_set.current_mesh().vertex_number() < 400
    assert mesh_set.current_mesh().face_number() > 0
    assert mesh_set.get_topological_measures()["unreferenced_vertices"] == 0


def test_empty_mesh_measures_and_cleanup():
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh())
    assert mesh_set.get_geometric_measures()["surface_area"] == 0
    assert mesh_set.get_topological_measures()["vertices_number"] == 0
    mesh_set.meshing_remove_null_faces()
    mesh_set.meshing_remove_duplicate_vertices()
    mesh_set.meshing_remove_unreferenced_vertices()
    assert mesh_set.current_mesh().vertex_number() == 0


def test_non_manifold_edge_is_reported():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]
    faces = [[0, 1, 2], [1, 0, 3], [0, 1, 4]]
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh(vertices, faces))
    measures = mesh_set.get_topological_measures()
    assert measures["non_two_manifold_edges"] == 1
    assert not measures["is_mesh_two_manifold"]


@pytest.mark.parametrize(
    "vertices,faces,error",
    [
        ([[0, 0, 0, 1]], [], ValueError),
        ([[0, 0, 0]], [0, 0, 0], ValueError),
        ([[0, 0, 0]], [[0.0, 0.0, 0.0]], TypeError),
        ([[np.nan, 0, 0]], [], ValueError),
    ],
)
def test_mesh_rejects_unsafe_ffi_inputs(vertices, faces, error):
    with pytest.raises(error):
        mojo.Mesh(vertices, faces)


def test_mesh_rejects_face_index_overflow_before_narrowing():
    faces = np.array([[0, 1, 2**63]], dtype=np.uint64)
    with pytest.raises(OverflowError):
        mojo.Mesh(np.zeros((3, 3)), faces)


def test_sampling_rejects_zero_area_mesh():
    mesh_set = mojo.MeshSet()
    mesh_set.add_mesh(mojo.Mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]]))
    with pytest.raises(ValueError, match="positive finite surface area"):
        mesh_set.generate_sampling_montecarlo(samplenum=1)


def test_border_geodesic_on_closed_mesh_is_infinite():
    mesh_set = mojo.MeshSet()
    mesh_set.create_tetrahedron()
    mesh_set.compute_scalar_by_border_distance_per_vertex()
    assert np.isinf(mesh_set.current_mesh().vertex_scalar_array()).all()
