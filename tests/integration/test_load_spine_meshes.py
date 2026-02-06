import numpy as np
import pytest
import trimesh
from numpy.testing import assert_allclose, assert_array_equal

from morph_spines.core.h5_schema import GRP_EDGES
from morph_spines.utils.morph_spine_loader import (
    load_morphology_with_spines,
    load_spine_meshes_for_morphology,
    load_spine_table,
)


@pytest.fixture
def vertices_ref():
    return [
        np.array([[0.5, 0.5, 0.5], [0.0, 1.6, 0.0], [1.0, 1.6, 0.0], [0.5, 1.6, 1.0]], dtype=float),
        np.array([[2.5, 2.0, 2.5], [2.0, 3.2, 2.0], [3.0, 3.2, 2.0], [2.5, 3.2, 3.0]], dtype=float),
        np.array([[3.5, 2.5, 3.0], [3.0, 3.8, 2.5], [4.0, 3.8, 2.5], [3.5, 3.8, 3.5]], dtype=float),
        np.array([[3.0, 2.0, 3.0], [2.5, 3.4, 2.5], [3.5, 3.4, 2.5], [3.0, 3.4, 3.5]], dtype=float),
    ]


@pytest.fixture
def centered_vertices_ref():
    return [
        np.array(
            [[0.0, 0.0, 0.0], [-0.5, 1.1, -0.5], [0.5, 1.1, -0.5], [0.0, 1.1, 0.5]], dtype=float
        ),
        np.array(
            [[0.0, 0.0, 0.0], [-0.5, 1.2, -0.5], [0.5, 1.2, -0.5], [0.0, 1.2, 0.5]], dtype=float
        ),
        np.array(
            [[0.0, 0.0, 0.0], [-0.5, 1.3, -0.5], [0.5, 1.3, -0.5], [0.0, 1.3, 0.5]], dtype=float
        ),
        np.array(
            [[0.0, 0.0, 0.0], [-0.5, 1.4, -0.5], [0.5, 1.4, -0.5], [0.0, 1.4, 0.5]], dtype=float
        ),
    ]


@pytest.fixture
def triangles_ref():
    return [
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int),
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int),
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int),
        np.array([[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=int),
    ]


@pytest.fixture
def meshes_reference(vertices_ref, triangles_ref):
    return [
        trimesh.Trimesh(vertices=vertices, faces=triangles)
        for vertices, triangles in zip(vertices_ref, triangles_ref, strict=True)
    ]


@pytest.fixture
def centered_meshes_reference(centered_vertices_ref, triangles_ref):
    return [
        trimesh.Trimesh(vertices=vertices, faces=triangles)
        for vertices, triangles in zip(centered_vertices_ref, triangles_ref, strict=True)
    ]


def assert_mesh_equal(mesh, expected_mesh):
    assert isinstance(mesh, trimesh.Trimesh)
    assert_allclose(mesh.vertices, expected_mesh.vertices)
    assert_array_equal(mesh.faces, expected_mesh.faces)


def test_load_spine_meshes_for_morphology(
    multiple_morph_spines_file, multiple_morph_ids, meshes_reference
):
    spine_table = load_spine_table(
        multiple_morph_spines_file, f"{GRP_EDGES}/{multiple_morph_ids[0]}"
    )

    meshes = load_spine_meshes_for_morphology(
        multiple_morph_spines_file,
        multiple_morph_ids[0],
        spines_are_centered=False,
        spine_table=spine_table,
    )

    assert len(meshes) == len(meshes_reference)
    for mesh, mesh_ref in zip(meshes, meshes_reference, strict=True):
        assert_mesh_equal(mesh, mesh_ref)


def test_load_spine_meshes_for_morphology_centered(
    multiple_morph_spines_centered_file, multiple_morph_ids, meshes_reference
):
    spine_table = load_spine_table(
        multiple_morph_spines_centered_file,
        f"{GRP_EDGES}/{multiple_morph_ids[0]}",
    )

    meshes = load_spine_meshes_for_morphology(
        multiple_morph_spines_centered_file,
        multiple_morph_ids[0],
        spines_are_centered=True,
        spine_table=spine_table,
    )

    assert len(meshes) == len(meshes_reference)
    for mesh, mesh_ref in zip(meshes, meshes_reference, strict=True):
        assert_mesh_equal(mesh, mesh_ref)


def test_load_spine_meshes_for_morphology_differed(single_morph_spines_file, meshes_reference):
    morphology_w_spines = load_morphology_with_spines(
        single_morph_spines_file, spines_are_centered=False, load_meshes=False
    )
    meshes = list(morphology_w_spines.spines.spine_meshes_for_morphology())

    assert len(meshes) == morphology_w_spines.spines.spine_count
    assert len(meshes) == len(meshes_reference)
    for mesh, mesh_ref in zip(meshes, meshes_reference, strict=True):
        assert_mesh_equal(mesh, mesh_ref)


def test_load_spine_meshes_for_morphology_centered_differed(
    single_morph_spines_centered_file, meshes_reference
):
    morphology_w_spines = load_morphology_with_spines(
        single_morph_spines_centered_file, spines_are_centered=True, load_meshes=False
    )
    meshes = list(morphology_w_spines.spines.spine_meshes_for_morphology())

    assert len(meshes) == morphology_w_spines.spines.spine_count
    assert len(meshes) == len(meshes_reference)
    for mesh, mesh_ref in zip(meshes, meshes_reference, strict=True):
        assert_mesh_equal(mesh, mesh_ref)


def test_load_centered_spine_meshes_for_morphology_centered_differed(
    single_morph_spines_centered_file, centered_meshes_reference
):
    morphology_w_spines = load_morphology_with_spines(
        single_morph_spines_centered_file, spines_are_centered=True, load_meshes=False
    )
    meshes = list(morphology_w_spines.spines.centered_spine_meshes_for_morphology())

    assert len(meshes) == morphology_w_spines.spines.spine_count
    assert len(meshes) == len(centered_meshes_reference)
    for mesh, mesh_ref in zip(meshes, centered_meshes_reference, strict=True):
        assert_mesh_equal(mesh, mesh_ref)


def test_load_spine_meshes_for_morphology_without_spine_table(
        single_morph_spines_file, single_morph_id, meshes_reference
):
    meshes = load_spine_meshes_for_morphology(
        single_morph_spines_file,
        single_morph_id,
        spines_are_centered=False,
        spine_table=None
    )

    assert len(meshes) == len(meshes_reference)
    for mesh, mesh_ref in zip(meshes, meshes_reference, strict=True):
        assert_mesh_equal(mesh, mesh_ref)