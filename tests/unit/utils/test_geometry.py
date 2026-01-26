import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from morph_spines.utils import geometry


@pytest.fixture
def spine_rotation_ref():
    return np.array(
        [0.18257419, 0.36514837, 0.54772256, 0.73029674],
        dtype=np.float64,
    )


@pytest.fixture
def spine_translation_ref():
    return np.array([0.111, 0.222, 0.333], dtype=np.float64)


@pytest.fixture
def spine_table(spine_rotation_ref, spine_translation_ref):
    return pd.DataFrame(
        [
            {
                "spine_rotation_x": spine_rotation_ref[0],
                "spine_rotation_y": spine_rotation_ref[1],
                "spine_rotation_z": spine_rotation_ref[2],
                "spine_rotation_w": spine_rotation_ref[3],
                "afferent_surface_x": spine_translation_ref[0],
                "afferent_surface_y": spine_translation_ref[1],
                "afferent_surface_z": spine_translation_ref[2],
            }
        ]
    )


@pytest.fixture
def spine_loc():
    return 0


@pytest.fixture
def spine_points_ref():
    return np.array([0.1, 0.2, 0.3], dtype=np.float64)


@pytest.fixture
def spine_points_transformed_ref():
    return np.array([0.211, 0.422, 0.633], dtype=np.float64)


def test_spine_transformations(spine_table, spine_loc, spine_rotation_ref, spine_translation_ref):
    spine_rotation, spine_translation = geometry.spine_transformations(spine_table, spine_loc)

    assert_allclose(spine_rotation_ref, spine_rotation.as_quat())
    assert_allclose(spine_translation_ref, spine_translation)


def test_transform_for_spine(
    spine_table, spine_loc, spine_points_ref, spine_points_transformed_ref
):
    spine_points_transformed = geometry.transform_for_spine(
        spine_table, spine_loc, spine_points_ref
    )

    assert_allclose(spine_points_transformed_ref, spine_points_transformed[0])


def test_inverse_transform_for_spine(
    spine_table, spine_loc, spine_points_ref, spine_points_transformed_ref
):
    spine_points = geometry.inverse_transform_for_spine(
        spine_table, spine_loc, spine_points_transformed_ref
    )

    assert_allclose(spine_points_ref, spine_points[0])
