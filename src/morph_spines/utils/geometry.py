"""Module to manipulate spatial objects.

Provides utilities to manipulate spatial objects, like skeletons and meshes belonging to neurons,
spines or soma.
"""

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
    spine_rotation = Rotation.from_quat(spine_row[COL_ROTATION].to_numpy(dtype=float))
    spine_transformation = spine_row[COL_TRANSLATION].to_numpy(dtype=float)

    return spine_rotation, spine_transformation


def transform_for_spine(
    spine_table: pd.DataFrame, spine_loc: int, spine_points: NDArray
) -> NDArray:
    """Apply spine coordinate system transformations.

    Apply the transformation from the local spine coordinate system
    to the global neuron coordinate system to a set of points.
    """
    spine_rotation, spine_transformation = spine_transformations(spine_table, spine_loc)
    return spine_rotation.apply(spine_points) + spine_transformation.reshape((1, -1))
