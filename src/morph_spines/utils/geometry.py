"""Module to manipulate spatial objects.

Provides utilities to manipulate spatial objects, like skeletons and meshes belonging to neurons,
spines or soma.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation


def transform_for_spine(
    spine_rotation: Rotation, spine_translation: NDArray, spine_points: NDArray
) -> NDArray:
    """Apply spine coordinate system transformations.

    Apply the transformation from the local spine coordinate system
    to the global neuron coordinate system to a set of points.
    """
    return spine_rotation.apply(spine_points) + spine_translation.reshape((1, -1))


def inverse_transform_for_spine(
    spine_rotation: Rotation, spine_translation: NDArray, spine_points: NDArray
) -> NDArray:
    """Apply spine coordinate system transformations.

    Apply the transformation from the global neuron coordinate system
    to the local spine coordinate system to a set of points.
    """
    return spine_rotation.inv().apply(spine_points - spine_translation.reshape((1, -1)))


def inverse_transform_matrix_for_spine(
    spine_rotation: Rotation, spine_translation: NDArray
) -> NDArray:
    """Get the spine inverse transform matrix from the spine table information."""
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = spine_rotation.inv().as_matrix()
    transform_matrix[:3, 3] = -spine_rotation.inv().apply(spine_translation)

    return transform_matrix
