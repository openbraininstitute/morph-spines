"""Utilities for mesh manipulation.

Provides helpers for filtering and extracting sub-meshes from triangle meshes,
including head/neck region classification for spine meshes.
"""

import numpy as np
from numpy.typing import NDArray


def filter_triangles_by_head_neck(
    triangles: NDArray,
    head_neck_offsets: NDArray,
    include_head: bool,
    include_neck: bool,
) -> list[NDArray]:
    """Filter triangles based on head/neck classification.

    Triangles within a spine are assumed to be sorted: neck triangles first,
    then head triangles. For branched spines (multiple heads), each head
    region is contiguous and follows the neck.

    head_neck_offsets is an offset-style array of H + 1 integers for a
    spine with H heads, where:

    - Neck triangles occupy triangles[0 : offsets[0]].
    - offsets[N] .. offsets[N+1] gives the triangle range for head N.
    - offsets[H] equals the total number of triangles.

    For a simple (single-head) spine this is a 2-element array
    [first_head_triangle, total_triangles].

    If head_neck_offsets is empty, all triangles are considered undefined
    and are always returned regardless of the filter flags.

    Args:
        triangles: All triangles for the spine.
        head_neck_offsets: Offset-style array of length H + 1 (H heads).
            Empty means all triangles are undefined.
        include_head: Whether to include head triangles.
        include_neck: Whether to include neck triangles.

    Returns:
        A list of NDArrays, ordered as: [neck, head_0, head_1, ...],
        filtered according to include_head and include_neck.
        When offsets are empty (undefined), returns a single-element list
        with all triangles.
    """
    filtered_triangles = []
    if len(head_neck_offsets) == 0:
        # No offsets: all triangles are undefined, always returned
        filtered_triangles.append(triangles)
        return filtered_triangles

    if include_neck:
        filtered_triangles.append(triangles[: head_neck_offsets[0]])

    if include_head:
        for i in range(len(head_neck_offsets) - 1):
            filtered_triangles.append(triangles[head_neck_offsets[i] : head_neck_offsets[i + 1]])

    return filtered_triangles


def submesh(vertices: NDArray, triangles: NDArray) -> tuple[NDArray, NDArray]:
    """Extract a submesh keeping only the vertices referenced by the given triangles.

    Args:
        vertices: Full vertex array of the mesh.
        triangles: Filtered triangle array (indices into vertices).

    Returns:
        A tuple (new_vertices, new_triangles) where new_vertices
        contains only the referenced vertices and new_triangles has
        remapped indices into new_vertices.
    """
    if len(triangles) == 0:
        return np.empty((0, 3), dtype=vertices.dtype), np.empty((0, 3), dtype=int)

    unique_indices, inverse = np.unique(triangles, return_inverse=True)
    new_vertices = vertices[unique_indices]
    new_triangles = inverse.reshape(triangles.shape)
    return new_vertices, new_triangles
