import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose
from scipy.spatial.transform import Rotation

from morph_spines.utils import geometry


@pytest.fixture
def spine_rotation_ref():
    return Rotation.from_quat(
        np.array([0.18257419, 0.36514837, 0.54772256, 0.73029674], dtype=np.float64)
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


def test_transform_for_spine(
    spine_rotation_ref, spine_translation_ref, spine_points_ref, spine_points_transformed_ref
):
    spine_points_transformed = geometry.transform_for_spine(
        spine_rotation_ref, spine_translation_ref, spine_points_ref
    )

    assert_allclose(spine_points_transformed_ref, spine_points_transformed[0])


def test_inverse_transform_for_spine(
    spine_rotation_ref, spine_translation_ref, spine_points_ref, spine_points_transformed_ref
):
    spine_points = geometry.inverse_transform_for_spine(
        spine_rotation_ref, spine_translation_ref, spine_points_transformed_ref
    )

    assert_allclose(spine_points_ref, spine_points[0])


def test_inverse_transform_matrix_for_spine(
    spine_rotation_ref, spine_translation_ref, spine_points_ref, spine_points_transformed_ref
):
    """The 4x4 inverse transform matrix should produce the same result as inverse_transform."""
    matrix = geometry.inverse_transform_matrix_for_spine(spine_rotation_ref, spine_translation_ref)

    # Apply the 4x4 matrix to the transformed point (in homogeneous coordinates)
    point_h = np.append(spine_points_transformed_ref, 1.0)
    result = matrix @ point_h

    assert_allclose(result[:3], spine_points_ref, atol=1e-10)


def test_inverse_transform_matrix_identity():
    """Identity rotation and zero translation should give an identity matrix."""
    rotation = Rotation.identity()
    translation = np.zeros(3)

    matrix = geometry.inverse_transform_matrix_for_spine(rotation, translation)

    assert_allclose(matrix, np.eye(4), atol=1e-10)


def test_transform_inverse_roundtrip(spine_rotation_ref, spine_translation_ref, spine_points_ref):
    """Applying transform then inverse should return the original points."""
    transformed = geometry.transform_for_spine(
        spine_rotation_ref, spine_translation_ref, spine_points_ref
    )
    recovered = geometry.inverse_transform_for_spine(
        spine_rotation_ref, spine_translation_ref, transformed
    )

    assert_allclose(recovered, spine_points_ref.reshape(1, -1), atol=1e-10)
