import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import h5py
import trimesh
from numpy.ma.testutils import assert_array_equal
from scipy.spatial.transform import Rotation

#from NeuroM.neurom.core.morphology import Neurite
from morph_spines.core.h5_schema import (
    COL_ROTATION, COL_TRANSLATION, COL_SPINE_MESH, COL_SPINE_ID,
    COL_AFF_SEC, GRP_SPINES, GRP_MESHES, GRP_VERTICES, GRP_TRIANGLES,
    GRP_OFFSETS
)

from morph_spines.core.spines import Spines


SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_FILE= f"{SAMPLE_DATA_DIR}/morph_with_spines_schema.h5"
MORPH_WITH_SPINES_ID = "01234"


@pytest.fixture
def spines():
    """Fixture providing a Spines instance"""
    spine_table = pd.read_hdf(SAMPLE_MORPH_WITH_SPINES_FILE, key=str(f"/edges/{MORPH_WITH_SPINES_ID}"))
    return Spines(
        meshes_filepath=SAMPLE_MORPH_WITH_SPINES_FILE,
        morphology_name=MORPH_WITH_SPINES_ID,
        spine_table=spine_table,
        centered_spine_skeletons=False,
        spines_are_centered=False
    )



def test_spine_count(spines):
    assert spines.spine_count == 2

def test_spine_transformations(spines):
    expected_transformations = (
        np.array([
            [0.44180798, -0.58406595, 0.68093515],
            [0.64080507, 0.73666259, 0.21609509],
            [-0.62783323, 0.34087416, 0.69973583]
        ]),
        np.array([0.926668, 0.0630049, 0.957736]))
    transformations = spines.spine_transformations(0)
    assert np.allclose(transformations[0].as_matrix(), expected_transformations[0], rtol=1e-6, atol=1e-8)
    assert np.allclose(transformations[1], expected_transformations[1], rtol=1e-6, atol=1e-8)

#def test_spine_skeletons(spines):
#    spine_skeletons = spines.spine_skeletons
#    assert len(spine_skeletons) == 2

def test__spine_mesh_points(spines):
    expected_points = np.array([[2, 2, 2,], [3, 2, 4,], [2, 3, 4], [1, 2, 4], [2, 1, 4]])
    points = spines._spine_mesh_points(spine_loc=0, transform=False)
    assert_array_equal(points, expected_points)

def test_spine_mesh_triangles(spines):
    expected_triangles = np.array([[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [1, 2, 3], [1, 3, 4]])
    triangles = spines.spine_mesh_triangles(spine_loc=0)
    assert_array_equal(triangles, expected_triangles)

def test_spine_mesh(spines):
    mesh = spines.spine_mesh(0)
    assert isinstance(mesh, trimesh.Trimesh)

def test_centered_spine_mesh(spines):
    mesh = spines.centered_spine_mesh(0)
    assert isinstance(mesh, trimesh.Trimesh)

#def test_spine_indices_for_section(spines):
#    spines.spine_indices_for_section(0)

#def spine_table_for_section(self, section_id):
#    spines.spine_table_for_section(0)
