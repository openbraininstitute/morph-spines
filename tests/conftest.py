"""Shared test fixtures and helpers for morph-spines tests."""

import h5py
import numpy as np
import pandas as pd
import pytest

from morph_spines.core.h5_schema import (
    GRP_MESHES,
    GRP_METADATA,
    GRP_MORPH,
    GRP_OFFSETS,
    GRP_SKELETONS,
    GRP_SPINES,
    GRP_TRIANGLES,
    GRP_VERTICES,
)

NEURON_NAME = "neuron_01"
SPINE_MORPH_NAME = "neuron_01"
N_SPINES = 3

# Morphology (in H5v1 format)
SAMPLE_POINTS = np.array(
    [[0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.4], [2.0, 0.0, 0.0, 0.3]],
    dtype=np.float32,
)
SAMPLE_STRUCTURE = np.array([[0, 3, -1]], dtype=np.int32)

# Mesh primitives
SAMPLE_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
SAMPLE_TRIANGLES = np.array([[0, 1, 2]], dtype=np.int32)

# Spine skeletons (3 root sections for 3 spines)
SKELETON_POINTS = np.array(
    [
        [0, 0, 0, 0.1],
        [0, 1, 0, 0.1],
        [0, 2, 0, 0.1],
        [0, 0, 1, 0.1],
        [0, 1, 1, 0.1],
        [0, 0, 2, 0.1],
        [0, 1, 2, 0.1],
    ],
    dtype=np.float32,
)
SKELETON_STRUCTURE = np.array([[0, 2, -1], [2, 2, -1], [4, 2, -1]], dtype=np.int32)

# Spine meshes (3 spines, each with its own [0,1,2] triangle into local vertices)
SPINE_MESH_VERTICES = np.tile(SAMPLE_VERTICES, (N_SPINES, 1))
SPINE_MESH_TRIANGLES = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]], dtype=np.int32)
SPINE_MESH_OFFSETS = np.array([[0, 0], [3, 1], [6, 2], [9, 3]], dtype=np.int32)

# Soma mesh
SOMA_VERTICES = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
SOMA_TRIANGLES = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)

# Edge/spine table column data
EDGE_TABLE_DATA = {
    "afferent_surface_x": np.array([1.0, 2.0, 3.0]),
    "afferent_surface_y": np.array([1.1, 2.1, 3.1]),
    "afferent_surface_z": np.array([1.2, 2.2, 3.2]),
    "afferent_center_x": np.array([0.5, 1.5, 2.5]),
    "afferent_center_y": np.array([0.6, 1.6, 2.6]),
    "afferent_center_z": np.array([0.7, 1.7, 2.7]),
    "spine_length": np.array([1.5, 2.0, 1.8]),
    "spine_orientation_vector_x": np.array([0.0, 0.0, 0.0]),
    "spine_orientation_vector_y": np.array([1.0, 1.0, 1.0]),
    "spine_orientation_vector_z": np.array([0.0, 0.0, 0.0]),
    "spine_rotation_x": np.array([0.0, 0.0, 0.0]),
    "spine_rotation_y": np.array([0.0, 0.0, 0.0]),
    "spine_rotation_z": np.array([0.0, 0.0, 0.0]),
    "spine_rotation_w": np.array([1.0, 1.0, 1.0]),
    "afferent_section_id": np.array([0, 0, 0], dtype=np.uint32),
    "afferent_segment_id": np.array([0, 0, 0], dtype=np.int32),
    "afferent_segment_offset": np.array([0.1, 0.2, 0.3]),
    "afferent_section_pos": np.array([0.1, 0.5, 0.9]),
}


@pytest.fixture
def valid_spine_table():
    """Create a minimal valid spine table DataFrame with all mandatory columns."""
    data = {col: arr.copy() for col, arr in EDGE_TABLE_DATA.items()}
    data["spine_morphology"] = [SPINE_MORPH_NAME] * N_SPINES
    data["spine_id"] = np.arange(N_SPINES, dtype=np.uint32)
    return pd.DataFrame(data)


@pytest.fixture
def sample_vertices():
    """Minimal 3-vertex mesh."""
    return SAMPLE_VERTICES.copy()


@pytest.fixture
def sample_triangles():
    """Single triangle referencing sample_vertices."""
    return SAMPLE_TRIANGLES.copy()


@pytest.fixture
def sample_points():
    """Minimal morphology points (3 samples, H5v1 format: x, y, z, radius)."""
    return SAMPLE_POINTS.copy()


@pytest.fixture
def sample_structure():
    """Minimal morphology structure (single root section)."""
    return SAMPLE_STRUCTURE.copy()


def write_minimal_valid_file(filepath):
    """Create a minimal valid morph-with-spines HDF5 file.

    Contains all four top-level groups (/morphology, /edges, /spines, /soma)
    with consistent data for a single neuron with 3 spines.
    """
    # Build spine table from shared constants
    spine_table = pd.DataFrame({col: arr.copy() for col, arr in EDGE_TABLE_DATA.items()})
    spine_table["spine_morphology"] = [SPINE_MORPH_NAME] * N_SPINES
    spine_table["spine_id"] = np.arange(N_SPINES, dtype=np.uint32)

    # /morphology
    with h5py.File(filepath, "w") as h5:
        morph_grp = h5.create_group(f"{GRP_MORPH}/{NEURON_NAME}")
        morph_grp.create_dataset("points", data=SAMPLE_POINTS)
        morph_grp.create_dataset("structure", data=SAMPLE_STRUCTURE)
        meta = morph_grp.create_group(GRP_METADATA)
        meta.attrs["version"] = np.array([1, 3], dtype=np.uint32)

    # /edges
    from morph_spines.utils.morph_spine_writer import write_spine_table

    write_spine_table(str(filepath), NEURON_NAME, spine_table)

    # /spines and /soma
    with h5py.File(filepath, "a") as h5:
        # /spines/skeletons
        skel_grp = h5.create_group(f"{GRP_SPINES}/{GRP_SKELETONS}/{SPINE_MORPH_NAME}")
        skel_grp.create_dataset("points", data=SKELETON_POINTS)
        skel_grp.create_dataset("structure", data=SKELETON_STRUCTURE)
        skel_meta = skel_grp.create_group(GRP_METADATA)
        skel_meta.attrs["version"] = np.array([1, 3], dtype=np.uint32)

        # /spines/meshes
        mesh_grp = h5.create_group(f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}")
        mesh_grp.create_dataset(GRP_VERTICES, data=SPINE_MESH_VERTICES, compression="gzip")
        mesh_grp.create_dataset(GRP_TRIANGLES, data=SPINE_MESH_TRIANGLES, compression="gzip")
        mesh_grp.create_dataset(GRP_OFFSETS, data=SPINE_MESH_OFFSETS, compression="gzip")

        # /soma
        soma_grp = h5.create_group(f"soma/{GRP_MESHES}/{NEURON_NAME}")
        soma_grp.create_dataset(GRP_VERTICES, data=SOMA_VERTICES)
        soma_grp.create_dataset(GRP_TRIANGLES, data=SOMA_TRIANGLES)


@pytest.fixture
def minimal_valid_file(tmp_path):
    """Create a minimal valid morph-with-spines file and return its path."""
    filepath = tmp_path / "valid.h5"
    write_minimal_valid_file(filepath)
    return filepath
