from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from morph_spines.core.soma import Soma

SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_v1.0.h5"
MORPH_WITH_SPINES_ID = "neuron_0"
EXPECTED_SOMA_VERTICES = np.array(
    [[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]]
)
EXPECTED_SOMA_TRIANGLES = np.array(
    [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]]
)


@pytest.fixture
def soma():
    """Fixture providing a Soma instance"""
    return Soma(meshes_filepath=SAMPLE_MORPH_WITH_SPINES_FILE, morphology_name=MORPH_WITH_SPINES_ID)


def test_soma_name(soma):
    assert soma.name == MORPH_WITH_SPINES_ID


def test_soma_center(soma):
    expected_center = np.array([0.0, 0.0, 0.0])
    assert_array_equal(soma.center, expected_center)


def test_soma_mesh(soma):
    assert_array_equal(soma.soma_mesh.vertices, EXPECTED_SOMA_VERTICES)
    assert_array_equal(soma.soma_mesh.faces, EXPECTED_SOMA_TRIANGLES)


def test_soma_mesh_points(soma):
    assert_array_equal(soma.soma_mesh_points, EXPECTED_SOMA_VERTICES)


def test_soma_mesh_triangles(soma):
    assert_array_equal(soma.soma_mesh_triangles, EXPECTED_SOMA_TRIANGLES)
