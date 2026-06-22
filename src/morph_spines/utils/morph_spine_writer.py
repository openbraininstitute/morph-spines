"""Writer functions for the morphology-with-spines file format.

Provides functions to write neuron spine data (spine tables, morphology,
soma meshes, spine meshes, and spine skeletons) to HDF5 files following
the morph-spines format specification.

Functions are ordered following the H5 file group structure:
    /edges          -> write_spine_table (+ validate_spine_table)
    /morphology     -> write_morphology
    /soma/meshes    -> write_soma_mesh
    /spines/meshes  -> write_spine_meshes
    /spines/skeletons -> write_spine_skeletons
"""

import os

import h5py
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from morph_spines.core.h5_schema import (
    ATT_VERSION,
    GRP_EDGES,
    GRP_MESHES,
    GRP_METADATA,
    GRP_MORPH,
    GRP_OFFSETS,
    GRP_SKELETONS,
    GRP_SOMA,
    GRP_SPINES,
    GRP_TRIANGLES,
    GRP_VERTICES,
    MANDATORY_COLUMNS,
    OPTIONAL_COLUMNS,
    SPINE_TABLE_VER_H5_DATASETS,
)


def validate_spine_table(spine_table: pd.DataFrame) -> None:
    """Validate that a spine table conforms to the morph-spines format specification.

    Checks:
        - All mandatory columns are present.
        - Column dtypes are compatible with the expected types.
        - No unknown columns (only mandatory + optional columns are allowed).

    Args:
        spine_table: The Pandas DataFrame to validate.

    Raises:
        ValueError: If any validation check fails. The error message lists all violations found.
    """
    errors: list[str] = []

    # Check mandatory columns are present
    missing = set(MANDATORY_COLUMNS.keys()) - set(spine_table.columns)
    if missing:
        errors.append(f"Missing mandatory columns: {sorted(missing)}")

    # Check for unknown columns
    known_columns = set(MANDATORY_COLUMNS.keys()) | set(OPTIONAL_COLUMNS.keys())
    unknown = set(spine_table.columns) - known_columns
    if unknown:
        errors.append(f"Unknown columns (not in spec): {sorted(unknown)}")

    # Check dtype compatibility for present columns
    all_columns = {**MANDATORY_COLUMNS, **OPTIONAL_COLUMNS}
    for col_name, expected_kind in all_columns.items():
        if col_name not in spine_table.columns:
            continue

        col = spine_table[col_name]
        actual_kind = col.dtype.kind

        if expected_kind == "str":
            # String columns: object dtype or string dtype
            if actual_kind not in ("O", "U", "S"):
                errors.append(f"Column '{col_name}': expected string type, got dtype '{col.dtype}'")
        elif expected_kind == "f":
            # Float columns: must be numeric (float or int, int will be cast)
            if actual_kind not in ("f", "i", "u"):
                errors.append(
                    f"Column '{col_name}': expected numeric (float) type, got dtype '{col.dtype}'"
                )
        elif expected_kind == "ui":
            # Unsigned int columns: accept int or float (pandas may upcast to float64)
            if actual_kind not in ("u", "i", "f"):
                errors.append(
                    f"Column '{col_name}': expected integer type, got dtype '{col.dtype}'"
                )
        elif expected_kind == "i":
            # Signed int columns: accept signed int or float (pandas may upcast to float64)
            if actual_kind not in ("i", "f"):
                errors.append(
                    f"Column '{col_name}': expected signed integer type, got dtype '{col.dtype}'"
                )
        else:
            raise ValueError(
                f"Unknown expected dtype kind '{expected_kind}' for column '{col_name}'"
            )

    # Check value constraints for numeric columns (skip if dtype is wrong)
    from morph_spines.utils.morph_spine_validator import check_column_values

    _CHECKED_COLUMNS = (
        "afferent_section_pos",
        "spine_length",
        "afferent_segment_offset",
        "afferent_segment_id",
        "spine_volume",
        "spine_neck_diameter",
    )
    for col_name in _CHECKED_COLUMNS:
        if col_name in spine_table.columns:
            if spine_table[col_name].dtype.kind in ("f", "i", "u"):
                for err in check_column_values(col_name, spine_table[col_name]):
                    errors.append(f"Column '{col_name}': {err.split(': ', 1)[1]}")

    if errors:
        raise ValueError("Spine table validation failed:\n  - " + "\n  - ".join(errors))


def write_spine_table(
    filepath: str,
    neuron_name: str,
    spine_table: pd.DataFrame,
) -> None:
    """Write a spine table to a morphology-with-spines H5 file.

    The spine table is stored column-wise as individual H5 datasets under
    /edges/{neuron_name}, following the v1.0 format specification. A metadata
    group with the format version is created alongside the datasets.

    Args:
        filepath: Path to the output H5 file. Created if it does not exist,
            appended to otherwise.
        neuron_name: Neuron name or spine collection name (used as the subgroup
            name under /edges).
        spine_table: Pandas DataFrame where each column becomes an H5 dataset.
            Column names must match the morph-spines format specification.

    Raises:
        ValueError: If the spine table fails validation (see validate_spine_table).
        ValueError: If the neuron_name group already exists under /edges.
    """
    validate_spine_table(spine_table)

    mode = "a" if os.path.exists(filepath) else "w"
    with h5py.File(filepath, mode) as h5:
        edges_grp = h5.require_group(GRP_EDGES)

        if neuron_name in edges_grp:
            raise ValueError(
                f"Group '/edges/{neuron_name}' already exists in {filepath}. "
                f"Cannot overwrite existing spine table."
            )

        neuron_grp = edges_grp.create_group(neuron_name)

        # Write version metadata
        metadata_grp = neuron_grp.create_group(GRP_METADATA)
        metadata_grp.attrs[ATT_VERSION] = np.array(SPINE_TABLE_VER_H5_DATASETS, dtype=np.uint32)

        # Write each column as a dataset
        for col_name in spine_table.columns:
            col_data = spine_table[col_name].to_numpy()

            # Convert object/string columns to variable-length UTF-8 strings
            if col_data.dtype == object or col_data.dtype.kind in ("U", "S"):
                col_data = np.array(col_data.astype(str), dtype=object)
                dt = h5py.string_dtype(encoding="utf-8")
                neuron_grp.create_dataset(col_name, data=col_data, dtype=dt)
            else:
                neuron_grp.create_dataset(col_name, data=col_data)


def write_morphology(
    filepath: str,
    neuron_name: str,
    points: NDArray,
    structure: NDArray,
    cell_family: int = 0,
) -> None:
    """Write a neuron morphology to a morphology-with-spines H5 file.

    Stores the morphology skeleton (points, structure) under
    /morphology/{neuron_name} following the H5 v1 morphology format.

    Args:
        filepath: Path to the output H5 file.
        neuron_name: Name of the neuron (used as the subgroup name under
            /morphology).
        points: Array of shape (N, 4) with [x, y, z, radius] per sample.
        structure: Array of shape (M, 3) with [offset, type, parent] per section.
        cell_family: Cell family identifier (default 0 = NEURON).

    Raises:
        ValueError: If the group already exists.
    """
    mode = "a" if os.path.exists(filepath) else "w"
    with h5py.File(filepath, mode) as h5:
        morph_grp = h5.require_group(GRP_MORPH)

        if neuron_name in morph_grp:
            raise ValueError(f"Group '/morphology/{neuron_name}' already exists in {filepath}.")

        neuron_grp = morph_grp.create_group(neuron_name)

        # Morphology metadata (H5 v1 format)
        metadata_grp = neuron_grp.create_group(GRP_METADATA)
        metadata_grp.attrs["cell_family"] = cell_family
        metadata_grp.attrs["version"] = np.array([1, 3], dtype=np.uint32)

        neuron_grp.create_dataset("points", data=points)
        neuron_grp.create_dataset("structure", data=structure)


def write_soma_mesh(
    filepath: str,
    neuron_name: str,
    vertices: NDArray,
    triangles: NDArray,
) -> None:
    """Write a soma mesh to a morphology-with-spines H5 file.

    Stores the soma mesh (vertices, triangles) under /soma/meshes/{neuron_name}.

    Args:
        filepath: Path to the output H5 file.
        neuron_name: Name of the neuron (used as the subgroup name under
            /soma/meshes).
        vertices: Array of shape (N, 3) with vertex coordinates.
        triangles: Array of shape (M, 3) with triangle vertex indices.

    Raises:
        ValueError: If the group already exists.
    """
    mode = "a" if os.path.exists(filepath) else "w"
    with h5py.File(filepath, mode) as h5:
        soma_grp = h5.require_group(GRP_SOMA)
        meshes_grp = soma_grp.require_group(GRP_MESHES)

        if neuron_name in meshes_grp:
            raise ValueError(f"Group '/soma/meshes/{neuron_name}' already exists in {filepath}.")

        neuron_grp = meshes_grp.create_group(neuron_name)
        neuron_grp.create_dataset(GRP_VERTICES, data=vertices)
        neuron_grp.create_dataset(GRP_TRIANGLES, data=triangles)


def write_spine_meshes(
    filepath: str,
    group_name: str,
    vertices: NDArray,
    triangles: NDArray,
    offsets: NDArray,
    head_neck_values: NDArray | None = None,
) -> None:
    """Write spine meshes to a morphology-with-spines H5 file.

    Stores the mesh data (vertices, triangles, offsets) under
    /spines/meshes/{group_name} with gzip compression.

    Args:
        filepath: Path to the output H5 file.
        group_name: Name of the mesh group (typically the neuron name or a
            spine library name matching 'spine_morphology' in the spine table).
        vertices: Array of shape (N, 3) with vertex coordinates.
        triangles: Array of shape (M, 3) with triangle vertex indices.
        offsets: Array of shape (num_spines + 1, 2) or (num_spines + 1, 3)
            with [vertex_offset, triangle_offset] pairs (and optionally
            head_neck_values offset).
        head_neck_values: Optional flat array of head/neck offset values.
            If provided, the offsets array must have 3 columns.

    Raises:
        ValueError: If the group already exists.
    """
    mode = "a" if os.path.exists(filepath) else "w"
    with h5py.File(filepath, mode) as h5:
        spines_grp = h5.require_group(GRP_SPINES)
        meshes_grp = spines_grp.require_group(GRP_MESHES)

        if group_name in meshes_grp:
            raise ValueError(f"Group '/spines/meshes/{group_name}' already exists in {filepath}.")

        mesh_grp = meshes_grp.create_group(group_name)
        mesh_grp.create_dataset(GRP_VERTICES, data=vertices, compression="gzip", compression_opts=5)
        mesh_grp.create_dataset(
            GRP_TRIANGLES, data=triangles, compression="gzip", compression_opts=5
        )
        mesh_grp.create_dataset(GRP_OFFSETS, data=offsets, compression="gzip", compression_opts=5)

        if head_neck_values is not None:
            mesh_grp.create_dataset(
                "head_neck_values",
                data=head_neck_values,
                compression="gzip",
                compression_opts=5,
            )


def write_spine_skeletons(
    filepath: str,
    group_name: str,
    points: NDArray,
    structure: NDArray,
) -> None:
    """Write spine skeletons to a morphology-with-spines H5 file.

    Stores the skeleton data (points, structure) under
    /spines/skeletons/{group_name} following the H5 v1 morphology format.

    Args:
        filepath: Path to the output H5 file.
        group_name: Name of the skeleton group (typically the neuron name
            matching 'spine_morphology' in the spine table).
        points: Array of shape (N, 4) with [x, y, z, radius] per sample.
        structure: Array of shape (M, 3) with [offset, type, parent] per section.

    Raises:
        ValueError: If the group already exists.
    """
    mode = "a" if os.path.exists(filepath) else "w"
    with h5py.File(filepath, mode) as h5:
        spines_grp = h5.require_group(GRP_SPINES)
        skeletons_grp = spines_grp.require_group(GRP_SKELETONS)

        if group_name in skeletons_grp:
            raise ValueError(
                f"Group '/spines/skeletons/{group_name}' already exists in {filepath}."
            )

        skel_grp = skeletons_grp.create_group(group_name)

        # Skeleton metadata (H5 v1 format)
        metadata_grp = skel_grp.create_group(GRP_METADATA)
        metadata_grp.attrs["cell_family"] = 0  # NEURON
        metadata_grp.attrs["version"] = np.array([1, 3], dtype=np.uint32)

        skel_grp.create_dataset("points", data=points)
        skel_grp.create_dataset("structure", data=structure)
