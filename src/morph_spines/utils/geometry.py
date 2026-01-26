"""Module to manipulate spatial objects.

Provides utilities to manipulate spatial objects, like skeletons and meshes belonging to neurons,
spines or soma.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from morph_spines.core.h5_schema import COL_ROTATION, COL_TRANSLATION


def spine_transformations(spine_table: pd.DataFrame, spine_loc: int) -> tuple[Rotation, NDArray]:
    """Spine coordinate system transformations.

    Transformations from the local coordinate system of a spine
    (origin near its root, y-axis pointing towards its tip) to the
    global coordinate system of the neuron.
    """
    spine_row = spine_table.loc[spine_loc]
    spine_rotation = Rotation.from_quat(np.array(spine_row[COL_ROTATION].to_numpy(dtype=float)))
    spine_translation = spine_row[COL_TRANSLATION].to_numpy(dtype=float)

    return spine_rotation, spine_translation


def transform_for_spine(
    spine_table: pd.DataFrame, spine_loc: int, spine_points: NDArray
) -> NDArray:
    """Apply spine coordinate system transformations.

    Apply the transformation from the local spine coordinate system
    to the global neuron coordinate system to a set of points.
    """
    spine_rotation, spine_translation = spine_transformations(spine_table, spine_loc)
    return spine_rotation.apply(spine_points) + spine_translation.reshape((1, -1))


def inverse_transform_for_spine(
    spine_table: pd.DataFrame, spine_loc: int, spine_points: NDArray
) -> NDArray:
    """Apply spine coordinate system transformations.

    Apply the transformation from the global neuron coordinate system
    to the local spine coordinate system to a set of points.
    """
    spine_rotation, spine_translation = spine_transformations(spine_table, spine_loc)

    return spine_rotation.inv().apply(spine_points - spine_translation.reshape((1, -1)))


def inverse_transform_matrix_for_spine(spine_table: pd.DataFrame, spine_loc: int) -> NDArray:
    """Get the spine inverse transform matrix from the spine table information."""
    rotation, translation = spine_transformations(spine_table, spine_loc)

    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation.inv().as_matrix()
    transform_matrix[:3, 3] = -rotation.inv().apply(translation)

    return transform_matrix
