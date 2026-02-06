import numpy as np
import pytest
from morphio import PointLevel, SectionType
from morphio.mut import Morphology
from neurom.core.morphology import Morphology as neuromMorphology
from numpy.testing import assert_array_equal


@pytest.fixture
def morphology_name():
    return "neuron_0"


@pytest.fixture
def morphology_points():
    return np.array(
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


@pytest.fixture
def morphology(morphology_name):
    """Fixture providing a Morphology instance"""
    # coll = morphio.Collection(SAMPLE_MORPH_WITH_SPINES_FILE)
    # morphology = coll.load(f"{GRP_MORPH}/{MORPH_WITH_SPINES_ID}")
    # return Morphology(morphology, MORPH_WITH_SPINES_ID, process_subtrees=False)

    morph = Morphology()
    morph.soma.points = [[0, 0, 0], [1, 1, 1]]
    morph.soma.diameters = [1, 1]

    section = morph.append_root_section(
        PointLevel([[2, 2, 2], [3, 2, 3]], [4, 4]), SectionType.axon
    )
    section.append_section(PointLevel([[3, 2, 3], [4, 3, 3]], [4, 4]))
    section.append_section(PointLevel([[3, 2, 3], [5, 5, 5]], [5, 5]))

    return neuromMorphology(morph.as_immutable(), morphology_name, process_subtrees=False)


def test_morphology_name(morphology, morphology_name):
    assert morphology.name == morphology_name


def test_morphology_npoints(morphology):
    expected_morph_npoints = 6
    assert morphology.to_morphio().n_points == expected_morph_npoints


def test_morphology_points(morphology, morphology_points):
    assert_array_equal(morphology.points, morphology_points)


def test_morphology_section_offsets(morphology):
    expected_section_offsets = np.array([0, 2, 4, 6], dtype=np.uint32)
    assert_array_equal(morphology.to_morphio().section_offsets, expected_section_offsets)


def test_morphology_soma_center(morphology):
    expected_soma_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    assert_array_equal(morphology.soma.center, expected_soma_center)
