from pathlib import Path

import morphio
import numpy as np
import pandas as pd
import pytest
import trimesh
from neurom import load_morphology as neurom_load_morphology
from numpy.ma.testutils import assert_array_equal

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


# def test_spine_skeletons(spines):
#    spine_skeletons = spines.spine_skeletons
#    assert len(spine_skeletons) == 2


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


# def test_spine_indices_for_section(spines):
#    spines.spine_indices_for_section(0)

# def spine_table_for_section(self, section_id):
#    spines.spine_table_for_section(0)
