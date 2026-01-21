from pathlib import Path

import morphio
import numpy as np
import pandas as pd
import pytest
import trimesh
from neurom import load_morphology as neurom_load_morphology
from neurom.core.morphology import Neurite
from numpy.ma.testutils import assert_array_equal
from numpy.testing import assert_allclose

from morph_spines.core.spines import Spines

SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_v1.0.h5"
SAMPLE_MORPH_WITH_SPINES_DATAFRAME_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_v0.1.h5"
MORPH_WITH_SPINES_ID = "neuron_0"


@pytest.fixture
def spines():
    """Fixture providing a Spines instance"""
    spine_table = pd.read_hdf(
        SAMPLE_MORPH_WITH_SPINES_DATAFRAME_FILE, key=str(f"/edges/{MORPH_WITH_SPINES_ID}")
    )
    coll = morphio.Collection(SAMPLE_MORPH_WITH_SPINES_DATAFRAME_FILE)
    spines_skeletons = neurom_load_morphology(
        coll.load(f"/spines/skeletons/{MORPH_WITH_SPINES_ID}")
    )

    return Spines(
        meshes_filepath=SAMPLE_MORPH_WITH_SPINES_DATAFRAME_FILE,
        morphology_name=MORPH_WITH_SPINES_ID,
        spine_table=spine_table,
        centered_spine_skeletons=spines_skeletons,
        spines_are_centered=True,
    )


@pytest.fixture
def spines_meshes(spines):
    """Fixture providing a Spines instance to test meshes"""
    spines.spine_table["afferent_section_id"] = [2, 2]
    return spines


def test_spine_count(spines):
    assert spines.spine_count == 2


def test_spine_transformations(spines):
    expected_transformations = (
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([2.0, 2.0, 3.0]),
    )
    transformations = spines.spine_transformations(0)
    assert len(transformations[0].as_matrix()) == len(expected_transformations[0])
    assert len(transformations[1]) == len(expected_transformations[1])

    assert np.allclose(
        transformations[0].as_matrix(),
        expected_transformations[0],  # , rtol=1e-5, atol=1e-7
    )
    assert np.allclose(transformations[1], expected_transformations[1])  # , rtol=1e-6, atol=1e-7)


def test__transform_spine_skeletons_fail_num_spines(spines):
    spines.spine_table.drop(index=0, inplace=True)

    with pytest.raises(ValueError):
        spines._transform_spine_skeletons()


def test_spine_skeletons(spines):
    spine_skeletons = spines.spine_skeletons
    expected_points = np.array([[2.0, 2.0, 3.0, 0.05], [2.0, 2.0, 5.0, 1.0]], dtype=np.float32)

    assert len(spine_skeletons) == 2
    assert isinstance(spine_skeletons[0], Neurite)
    assert spine_skeletons[0].length == 2.0
    assert_allclose(spine_skeletons[0].points, expected_points)


def test_centered_spine_skeletons(spines):
    spine_skeletons = spines.centered_spine_skeletons
    expected_points = np.array([[0.0, 0.0, 0.0, 0.05], [0.0, 0.0, 2.0, 1.0]], dtype=np.float32)

    assert len(spine_skeletons) == 2
    assert isinstance(spine_skeletons[0], Neurite)
    assert spine_skeletons[0].length == 2.0
    assert_allclose(spine_skeletons[0].points, expected_points)


def test__spine_mesh_points(spines):
    expected_points = np.array(
        [
            [0.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [-1.0, 0.0, 1.0],
            [0.0, -1.0, 1.0],
        ]
    )
    points = spines._spine_mesh_points(spine_loc=0, transform=False)
    assert_array_equal(points, expected_points)


def test_spine_mesh_triangles(spines):
    expected_triangles = np.array(
        [[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [1, 2, 3], [1, 3, 4]]
    )
    triangles = spines.spine_mesh_triangles(spine_loc=0)
    assert_array_equal(triangles, expected_triangles)


def test_spine_mesh(spines):
    mesh = spines.spine_mesh(0)
    assert isinstance(mesh, trimesh.Trimesh)


def test_centered_spine_mesh(spines):
    mesh = spines.centered_spine_mesh(0)
    assert isinstance(mesh, trimesh.Trimesh)


def test_spine_indices_for_section(spines):
    expected_indices = [0]
    indices = spines.spine_indices_for_section(2)

    assert_array_equal(expected_indices, indices)


def test_spine_table_for_section(spines):
    spine_table = spines.spine_table_for_section(2)

    assert isinstance(spine_table, pd.DataFrame)
    assert len(spine_table.columns) == 20
    assert set(
        [
            "afferent_surface_x",
            "afferent_center_x",
            "spine_length",
            "spine_orientation_vector_x",
            "spine_rotation_x",
            "afferent_section_id",
        ]
    ).issubset(set(spine_table.columns))
    assert spine_table.loc[0, "afferent_surface_x"] == np.float64(2.0)
    assert spine_table.loc[0, "spine_length"] == np.float64(2.0)


def test_spine_meshes_for_section(spines_meshes):
    meshes = list(spines_meshes.spine_meshes_for_section(2))
    expected_points = [
        [2.0, 2.0, 2.0],
        [3.0, 2.0, 4.0],
        [2.0, 3.0, 4.0],
        [1.0, 2.0, 4.0],
        [2.0, 1.0, 4.0],
    ]

    assert len(meshes) == 2
    assert isinstance(meshes[0], trimesh.Trimesh)
    assert_allclose(meshes[0].vertices, expected_points)


def test_compound_spine_mesh_for_section(spines_meshes):
    mesh = spines_meshes.compound_spine_mesh_for_section(2)
    expected_points = [5.0, 5.0, 5.0]

    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) == 9
    assert_allclose(mesh.vertices[5], expected_points)


def test_centered_spine_meshes_for_section(spines_meshes):
    meshes = list(spines_meshes.centered_spine_meshes_for_section(2))
    expected_points = [
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [-1.0, 0.0, 1.0],
        [0.0, -1.0, 1.0],
    ]

    assert len(meshes) == 2
    assert isinstance(meshes[0], trimesh.Trimesh)
    assert_allclose(meshes[0].vertices, expected_points)


def test_compound_centered_spine_mesh_for_section(spines_meshes):
    mesh = spines_meshes.compound_centered_spine_mesh_for_section(2)
    expected_points = [0.0, 0.0, -2.0]

    assert isinstance(mesh, trimesh.Trimesh)
    assert len(mesh.vertices) == 9
    assert_allclose(mesh.vertices[5], expected_points)
