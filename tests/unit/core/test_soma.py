# from pathlib import Path

import h5py
import numpy as np
import pytest
import trimesh
from numpy.testing import assert_array_equal

from morph_spines.core.h5_schema import (
    GRP_MESHES,
    GRP_SOMA,
    GRP_TRIANGLES,
    GRP_VERTICES,
)
from morph_spines.core.soma import Soma


@pytest.fixture
def morphology_name():
    return "neuron_0"


@pytest.fixture
def soma_vertices():
    return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])


@pytest.fixture
def soma_triangles():
    return np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]]
    )


@pytest.fixture
def soma_filename(tmp_path, morphology_name, soma_vertices, soma_triangles):
    f = tmp_path / "soma_file.h5"

    with h5py.File(f, "w") as h5:
        soma_grp = h5.create_group(GRP_SOMA)
        mesh_grp = soma_grp.create_group(GRP_MESHES)
        morph_grp = mesh_grp.create_group(morphology_name)
        morph_grp.create_dataset(GRP_VERTICES, data=soma_vertices)
        morph_grp.create_dataset(GRP_TRIANGLES, data=soma_triangles)

    return f


@pytest.fixture
def soma(soma_filename, morphology_name):
    """Fixture providing a Soma instance"""
    return Soma(meshes_filepath=soma_filename, morphology_name=morphology_name)


def test_soma_name(soma, morphology_name):
    assert soma.name == morphology_name


def test_soma_center(soma, soma_vertices):
    expected_center = soma_vertices.mean(axis=0)
    assert_array_equal(soma.center, expected_center)


def test_soma_mesh(soma, soma_vertices, soma_triangles):
    soma_mesh = soma.soma_mesh

    assert isinstance(soma_mesh, trimesh.Trimesh)
    assert_array_equal(soma_mesh.vertices, soma_vertices)
    assert_array_equal(soma_mesh.faces, soma_triangles)


def test_soma_mesh_points(soma, soma_vertices):
    assert_array_equal(soma.soma_mesh_points, soma_vertices)


def test_soma_mesh_triangles(soma, soma_triangles):
    assert_array_equal(soma.soma_mesh_triangles, soma_triangles)


def test_soma_missing_group_raises(tmp_path):
    """Accessing soma mesh when the H5 group doesn't exist raises ValueError."""
    f = tmp_path / "empty.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("other_group")

    soma = Soma(meshes_filepath=str(f), morphology_name="nonexistent")
    with pytest.raises(ValueError, match="Soma mesh not found"):
        _ = soma.soma_mesh_points


def test_soma_invalid_file_raises(tmp_path):
    """Accessing soma mesh with a non-existent file raises ValueError."""
    soma = Soma(meshes_filepath=str(tmp_path / "does_not_exist.h5"), morphology_name="neuron_0")
    with pytest.raises(ValueError, match="Cannot open file"):
        _ = soma.soma_mesh_points


def test_soma_caches_data(soma, soma_vertices, soma_triangles):
    """Verify that repeated access uses cached data (no re-read)."""
    # First access triggers load
    _ = soma.soma_mesh_points
    # Second access should use cache
    assert_array_equal(soma.soma_mesh_points, soma_vertices)
    assert_array_equal(soma.soma_mesh_triangles, soma_triangles)
