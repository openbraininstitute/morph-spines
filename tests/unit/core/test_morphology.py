from pathlib import Path

import morphio
import numpy as np
import pytest
from neurom.core.morphology import Morphology
from numpy.testing import assert_array_equal

from morph_spines.core.h5_schema import GRP_MORPH

SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_v1.0.h5"
MORPH_WITH_SPINES_ID = "neuron_0"


@pytest.fixture
def morphology():
    """Fixture providing a Morphology instance"""
    coll = morphio.Collection(SAMPLE_MORPH_WITH_SPINES_FILE)
    morphology = coll.load(f"{GRP_MORPH}/{MORPH_WITH_SPINES_ID}")
    return Morphology(morphology, MORPH_WITH_SPINES_ID, process_subtrees=False)


def test_morphology_name(morphology):
    assert morphology.name == MORPH_WITH_SPINES_ID


def test_morphology_npoints(morphology):
    expected_morph_npoints = 6
    assert morphology.to_morphio().n_points == expected_morph_npoints


def test_morphology_points(morphology):
    expected_morph_points = np.array(
        [
            [2.0, 2.0, 2.0, 2.0],
            [3.0, 2.0, 3.0, 2.0],
            [3.0, 2.0, 3.0, 2.0],
            [4.0, 3.0, 3.0, 2.0],
            [3.0, 2.0, 3.0, 2.5],
            [5.0, 5.0, 5.0, 2.5],
        ],
        dtype=np.float32,
    )
    assert_array_equal(morphology.points, expected_morph_points)


def test_morphology_section_offsets(morphology):
    expected_section_offsets = np.array([0, 2, 4, 6], dtype=np.uint32)
    assert_array_equal(morphology.to_morphio().section_offsets, expected_section_offsets)


def test_morphology_soma_center(morphology):
    expected_soma_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert_array_equal(morphology.soma.center, expected_soma_center)
