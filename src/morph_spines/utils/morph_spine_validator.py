"""Validates the structure and data integrity of a morphology-with-spines HDF5 file.

This module provides a public API function validate_morph_with_spines_file() that
checks whether a given HDF5 file conforms to the morph-spines format specification.

Structure checks (always performed):
    - File exists, is a file, and is a valid HDF5.
    - Mandatory top-level groups present: '/morphology', '/edges' and '/spines'.
    - /morphology/{name}: is a group with 'points' and 'structure' datasets.
    - /edges/{name}: is a group with 'metadata' subgroup containing 'version'
      attribute; 'version' is (1, 0); all mandatory column datasets present
      (spines table); spine_morphology references existing /spines/skeletons
      groups.
    - /spines/skeletons/{name}: is a group with 'points' and 'structure' datasets.
    - /spines/meshes/{name} (optional): if present, it contains 'vertices', 'triangles'
      and 'offsets' datasets.
    - /soma/meshes/{name} (optional): if present, it contains 'vertices' and 'triangles'
      datasets.
    - Warnings for unexpected top-level groups or unknown spine table columns.

Data integrity checks (when check_data_integrity=True):
    - /morphology points: shape (N, 4), float32 dtype, no NaN/Inf, non-empty.
    - /morphology structure: shape (M, 3), non-empty.
    - /edges column dtypes match the spec of the spine table (float, int, string).
    - /edges float columns: no NaN/Inf values.
    - /edges dataset lengths are consistent (all same size).
    - /edges afferent_section_pos values in [0, 1].
    - /edges spine_length values > 0.
    - /edges afferent_segment_offset values >= 0.
    - /edges afferent_segment_id values >= 0.
    - /edges spine_type values are valid SpineType enum values (if present).
    - /edges spine_volume values > 0 (if present).
    - /edges spine_neck_diameter values > 0 (if present).
    - /spines/skeletons points: shape (N, 4), float32 dtype, no NaN/Inf, non-empty.
    - /spines/skeletons structure: shape (M, 3), non-empty.
    - /spines/meshes vertices: shape (N, 3), no NaN/Inf.
    - /spines/meshes triangles: shape (M, 3), no negative indices.
    - /spines/meshes offsets: shape (K, 2|3), non-decreasing, first row is (0, 0).
    - /spines/meshes per-spine triangle indices < local vertex count (via offsets).
    - /soma/meshes vertices: shape (N, 3), no NaN/Inf.
    - /soma/meshes triangles: shape (M, 3), no negative indices,
      indices < vertex count.
    - Cross-group: /edges neuron names subgroups exist in /morphology.
    - Cross-group: spine_morphology values reference existing /spines/meshes groups.
    - Cross-group: spine_id values are valid indices into skeleton root sections
      and mesh offsets.
    - Cross-group: afferent_section_id values are valid section indices in
      /morphology/{name}/structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from morph_spines.core.h5_schema import (
    ATT_VERSION,
    GRP_EDGES,
    GRP_HEAD_NECK_VALUES,
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


def check_column_values(col_name: str, data) -> list[str]:
    """Check numeric column values for known constraints.

    Args:
        col_name: Name of the spine table column.
        data: Array-like of values (numpy array or pandas Series).

    Returns:
        A list of error message strings. Empty if all values are valid.
    """
    arr = np.asarray(data)
    errors: list[str] = []

    if col_name == "afferent_section_pos":
        n = int(np.sum((arr < 0) | (arr > 1)))
        if n > 0:
            errors.append(f"{col_name}: {n} values not in [0, 1]")
    elif col_name == "spine_length":
        n = int(np.sum(arr <= 0))
        if n > 0:
            errors.append(f"{col_name}: {n} values not > 0")
    elif col_name in ("afferent_segment_offset", "afferent_segment_id"):
        n = int(np.sum(arr < 0))
        if n > 0:
            errors.append(f"{col_name}: {n} values not >= 0")
    elif col_name in ("spine_volume", "spine_neck_diameter"):
        n = int(np.sum(arr <= 0))
        if n > 0:
            errors.append(f"{col_name}: {n} values not > 0")

    return errors


@dataclass
class ValidationResult:
    """Container for validation results.

    Attributes:
        is_valid: True if no errors were found (warnings are allowed).
        data_integrity_checked: Whether data-level integrity checks were performed
            in addition to the structural validation.
        errors: Critical issues that make the file non-conformant.
        warnings: Non-critical issues or deviations from best practices.
        info: Informational messages about what was found.
    """

    is_valid: bool = True
    data_integrity_checked: bool = False
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        """Record a validation error and mark the result as invalid."""
        self.errors.append(msg)
        self.is_valid = False

    def add_warning(self, msg: str) -> None:
        """Record a non-critical warning."""
        self.warnings.append(msg)

    def add_info(self, msg: str) -> None:
        """Record an informational message."""
        self.info.append(msg)

    def merge(self, other: ValidationResult) -> None:
        """Merge another ValidationResult into this one."""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        if not other.is_valid:
            self.is_valid = False

    def __str__(self) -> str:
        """Return a human-readable summary of the validation result."""
        status = "VALID" if self.is_valid else "INVALID"
        scope = "structure + data integrity" if self.data_integrity_checked else "structure only"
        lines = [f"Validation result: {status} (checked: {scope})"]
        if self.info:
            lines.append(f"  Info ({len(self.info)}):")
            for msg in self.info:
                lines.append(f"    - {msg}")
        if self.warnings:
            lines.append(f"  Warnings ({len(self.warnings)}):")
            for msg in self.warnings:
                lines.append(f"    - {msg}")
        if self.errors:
            lines.append(f"  Errors ({len(self.errors)}):")
            for msg in self.errors:
                lines.append(f"    - {msg}")
        return "\n".join(lines)


def validate_morph_with_spines_file(
    filepath: str | Path,
    check_data_integrity: bool = False,
) -> ValidationResult:
    """Validate a morphology-with-spines HDF5 file.

    By default, checks only file structure: the presence and nesting of expected HDF5
    groups, datasets, and metadata attributes. When check_data_integrity is True,
    also validates dataset shapes, dtypes, value ranges, and cross-group consistency.

    Args:
        filepath: Path to the HDF5 file to validate.
        check_data_integrity: If True, perform additional data-level checks such as
            dataset shapes, dtype compatibility, value ranges, NaN/Inf detection,
            and cross-group referential integrity.

    Returns:
        A ValidationResult with errors, warnings, and information messages.
    """
    filepath = Path(filepath)
    result = ValidationResult(data_integrity_checked=check_data_integrity)

    # Basic file checks
    if not filepath.exists():
        result.add_error(f"File not found: {filepath}")
        return result

    if not filepath.is_file():
        result.add_error(f"Path is not a file: {filepath}")
        return result

    # File structure checks
    try:
        h5 = h5py.File(filepath, "r")
    except OSError as e:
        result.add_error(f"Cannot open file as HDF5: {e}")
        return result

    with h5:
        # Validate top-level groups
        top_keys = set(h5.keys())

        # /morphology is mandatory
        if GRP_MORPH not in top_keys:
            result.add_error(f"Missing mandatory top-level group: /{GRP_MORPH}")
        else:
            result.merge(_validate_morphology_group(h5[GRP_MORPH], check_data_integrity))

        # /edges is mandatory
        if GRP_EDGES not in top_keys:
            result.add_error(f"Missing mandatory top-level group: /{GRP_EDGES}")
        else:
            result.merge(_validate_edges_group(h5[GRP_EDGES], check_data_integrity))

        # /spines is mandatory (skeletons required, meshes optional)
        if GRP_SPINES not in top_keys:
            result.add_error(f"Missing mandatory top-level group: /{GRP_SPINES}")
        else:
            result.merge(_validate_spines_group(h5[GRP_SPINES], check_data_integrity))

        # /soma is optional
        if GRP_SOMA in top_keys:
            result.merge(_validate_soma_group(h5[GRP_SOMA], check_data_integrity))
        else:
            result.add_info(f"Optional group /{GRP_SOMA} not present")

        # Check for unexpected top-level groups
        expected_top = {GRP_EDGES, GRP_MORPH, GRP_SOMA, GRP_SPINES}
        unexpected = top_keys - expected_top
        if unexpected:
            result.add_warning(f"Unexpected top-level groups: {sorted(unexpected)}")

        # Cross-group integrity (only with data integrity checks)
        if check_data_integrity and result.is_valid:
            result.merge(_validate_cross_group_integrity(h5))

    return result


def _validate_h5v1_morphology_subgroup(
    grp: h5py.Group, path: str, check_data_integrity: bool, result: ValidationResult
) -> None:
    """Validate an H5v1 morphology subgroup (points, structure, metadata).

    This is the common structure shared by /morphology/{name} and
    /spines/skeletons/{name} groups.

    Args:
        grp: The HDF5 group to validate.
        path: Display path for error messages (e.g. "/morphology/neuron_01").
        check_data_integrity: Whether to check dataset shapes, dtypes, and values.
        result: ValidationResult to accumulate findings into.
    """
    keys = set(grp.keys())

    if "points" not in keys:
        result.add_error(f"{path}: missing 'points' dataset")
    if "structure" not in keys:
        result.add_error(f"{path}: missing 'structure' dataset")
    if GRP_METADATA not in keys:
        result.add_warning(f"{path}: missing '{GRP_METADATA}' subgroup")

    if check_data_integrity:
        if "points" in keys:
            points = grp["points"]
            if not isinstance(points, h5py.Dataset):
                result.add_error(f"{path}/points is not a dataset")
            elif points.ndim != 2 or points.shape[1] != 4:
                result.add_error(f"{path}/points: expected shape (N, 4), got {points.shape}")
            elif points.shape[0] == 0:
                result.add_error(f"{path}/points: dataset is empty")
            else:
                if points.dtype != np.float32:
                    result.add_warning(f"{path}/points: expected float32, got {points.dtype}")
                data = points[:]
                if np.any(np.isnan(data)):
                    result.add_error(f"{path}/points: contains NaN values")
                if np.any(np.isinf(data)):
                    result.add_error(f"{path}/points: contains Inf values")

        if "structure" in keys:
            structure = grp["structure"]
            if not isinstance(structure, h5py.Dataset):
                result.add_error(f"{path}/structure is not a dataset")
            elif structure.ndim != 2 or structure.shape[1] != 3:
                result.add_error(f"{path}/structure: expected shape (M, 3), got {structure.shape}")
            elif structure.shape[0] == 0:
                result.add_error(f"{path}/structure: dataset is empty")


def _validate_morphology_group(
    morph_grp: h5py.Group, check_data_integrity: bool
) -> ValidationResult:
    """Validate the /morphology group structure and data."""
    result = ValidationResult()

    neuron_names = list(morph_grp.keys())
    if len(neuron_names) == 0:
        result.add_error("/morphology group is empty (no neuron subgroups)")
        return result

    result.add_info(f"/morphology contains {len(neuron_names)} neuron(s): {neuron_names}")

    for name in neuron_names:
        item = morph_grp[name]
        if not isinstance(item, h5py.Group):
            result.add_error(f"/morphology/{name} is not a group")
            continue

        _validate_h5v1_morphology_subgroup(
            item, f"/morphology/{name}", check_data_integrity, result
        )

    return result


def _validate_edges_group(edges_grp: h5py.Group, check_data_integrity: bool) -> ValidationResult:
    """Validate the /edges group structure and data."""
    result = ValidationResult()

    neuron_names = list(edges_grp.keys())
    if len(neuron_names) == 0:
        result.add_error("/edges group is empty (no neuron subgroups)")
        return result

    result.add_info(f"/edges contains {len(neuron_names)} spine table(s): {neuron_names}")

    for name in neuron_names:
        item = edges_grp[name]
        if not isinstance(item, h5py.Group):
            result.add_error(f"/edges/{name} is not a group")
            continue

        keys = set(item.keys())

        # Metadata subgroup is mandatory
        if GRP_METADATA not in keys:
            result.add_error(f"/edges/{name}: missing '{GRP_METADATA}' subgroup")
        else:
            metadata = item[GRP_METADATA]
            if not isinstance(metadata, h5py.Group):
                result.add_error(f"/edges/{name}/{GRP_METADATA} is not a group")
            elif ATT_VERSION not in metadata.attrs:
                result.add_error(f"/edges/{name}/{GRP_METADATA}: missing '{ATT_VERSION}' attribute")
            else:
                version = tuple(metadata.attrs[ATT_VERSION])
                if version != SPINE_TABLE_VER_H5_DATASETS:
                    result.add_error(
                        f"/edges/{name}: unsupported spine table version {version}, "
                        f"expected {SPINE_TABLE_VER_H5_DATASETS}"
                    )

        # Check mandatory columns are present as datasets
        dataset_keys = {k for k in keys if k != GRP_METADATA}
        missing_cols = set(MANDATORY_COLUMNS.keys()) - dataset_keys
        if missing_cols:
            result.add_error(
                f"/edges/{name}: missing mandatory column datasets: {sorted(missing_cols)}"
            )

        # Check for unknown columns
        known_cols = set(MANDATORY_COLUMNS.keys()) | set(OPTIONAL_COLUMNS.keys())
        unknown_cols = dataset_keys - known_cols
        if unknown_cols:
            result.add_warning(
                f"/edges/{name}: unknown column datasets (not in spec): {sorted(unknown_cols)}"
            )

        # Check spine_morphology references existing /spines/skeletons groups
        if "spine_morphology" in dataset_keys:
            spine_morph_dset = item["spine_morphology"]
            if isinstance(spine_morph_dset, h5py.Dataset):
                h5_file = item.file
                skeletons_path = f"{GRP_SPINES}/{GRP_SKELETONS}"
                if skeletons_path in h5_file:
                    skeleton_names = set(h5_file[skeletons_path].keys())
                    values = set(spine_morph_dset[:].astype(str))
                    missing = values - skeleton_names
                    if missing:
                        result.add_error(
                            f"/edges/{name}/spine_morphology: "
                            f"references groups not in "
                            f"/{skeletons_path}: {sorted(missing)}"
                        )

        if check_data_integrity:
            _validate_edges_data_integrity(item, name, result)

    return result


def _validate_edges_data_integrity(
    spine_table_grp: h5py.Group, name: str, result: ValidationResult
) -> None:
    """Validate data integrity of datasets within an /edges/{name} group."""
    all_columns = {**MANDATORY_COLUMNS, **OPTIONAL_COLUMNS}
    dataset_keys = {k for k in spine_table_grp.keys() if k != GRP_METADATA}

    # All datasets must have the same length
    lengths = {}
    for col_name in dataset_keys:
        item = spine_table_grp[col_name]
        if not isinstance(item, h5py.Dataset):
            result.add_error(f"/edges/{name}/{col_name} is not a dataset")
            continue
        if item.ndim > 1:
            result.add_error(f"/edges/{name}/{col_name}: must be 1D, got shape {item.shape}")
            continue
        length = item.shape[0] if item.shape != () else 1
        lengths[col_name] = length

    unique_lengths = set(lengths.values())
    if len(unique_lengths) > 1:
        result.add_error(f"/edges/{name}: datasets have inconsistent lengths: {lengths}")

    # Check dtypes for known columns
    for col_name in dataset_keys:
        if col_name not in all_columns:
            continue
        item = spine_table_grp[col_name]
        if not isinstance(item, h5py.Dataset):
            continue

        expected_kind = all_columns[col_name]
        actual_kind = item.dtype.kind

        if expected_kind == "str":
            if actual_kind not in ("O", "U", "S"):
                result.add_error(
                    f"/edges/{name}/{col_name}: expected string type, got dtype '{item.dtype}'"
                )
        elif expected_kind == "f":
            if actual_kind not in ("f", "i", "u"):
                result.add_error(
                    f"/edges/{name}/{col_name}: expected numeric type, got dtype '{item.dtype}'"
                )
            else:
                data = item[:]
                if np.any(np.isnan(data)):
                    result.add_error(f"/edges/{name}/{col_name}: contains NaN values")
                if np.any(np.isinf(data)):
                    result.add_error(f"/edges/{name}/{col_name}: contains Inf values")
        elif expected_kind in ("i", "ui"):
            if actual_kind not in ("i", "u", "f"):
                result.add_error(
                    f"/edges/{name}/{col_name}: expected integer type, got dtype '{item.dtype}'"
                )

    # Validate column value ranges
    _CHECKED_COLUMNS = (
        "afferent_section_pos",
        "spine_length",
        "afferent_segment_offset",
        "afferent_segment_id",
        "spine_volume",
        "spine_neck_diameter",
    )
    for col_name in _CHECKED_COLUMNS:
        if col_name in dataset_keys:
            data = spine_table_grp[col_name][:]
            for err in check_column_values(col_name, data):
                result.add_error(f"/edges/{name}/{err}")

    # Validate spine_type contains only known values (optional column)
    if "spine_type" in dataset_keys:
        from morph_spines.core.spine_type import SpineType

        valid_types = {t.value for t in SpineType}
        actual_types = set(spine_table_grp["spine_type"][:].astype(str))
        invalid_types = actual_types - valid_types
        if invalid_types:
            result.add_error(f"/edges/{name}/spine_type: unknown values: {sorted(invalid_types)}")


def _validate_spines_group(spines_grp: h5py.Group, check_data_integrity: bool) -> ValidationResult:
    """Validate the /spines group structure and data."""
    result = ValidationResult()

    keys = set(spines_grp.keys())

    # /spines/skeletons is mandatory
    if GRP_SKELETONS not in keys:
        result.add_error(f"/spines: missing mandatory subgroup '{GRP_SKELETONS}'")
    else:
        result.merge(_validate_spines_skeletons(spines_grp[GRP_SKELETONS], check_data_integrity))

    # /spines/meshes is optional
    if GRP_MESHES in keys:
        result.merge(_validate_spines_meshes(spines_grp[GRP_MESHES], check_data_integrity))
    else:
        result.add_info("/spines/meshes not present (optional)")

    # Unexpected subgroups
    expected = {GRP_SKELETONS, GRP_MESHES}
    unexpected = keys - expected
    if unexpected:
        result.add_warning(f"/spines: unexpected subgroups: {sorted(unexpected)}")

    return result


def _validate_spines_skeletons(
    skeletons_grp: h5py.Group, check_data_integrity: bool
) -> ValidationResult:
    """Validate /spines/skeletons subgroups."""
    result = ValidationResult()

    group_names = list(skeletons_grp.keys())
    if len(group_names) == 0:
        result.add_error("/spines/skeletons is empty (no skeleton groups)")
        return result

    result.add_info(f"/spines/skeletons contains {len(group_names)} group(s): {group_names}")

    for name in group_names:
        item = skeletons_grp[name]
        if not isinstance(item, h5py.Group):
            result.add_error(f"/spines/skeletons/{name} is not a group")
            continue

        _validate_h5v1_morphology_subgroup(
            item, f"/spines/skeletons/{name}", check_data_integrity, result
        )

    return result


def _validate_spines_meshes(meshes_grp: h5py.Group, check_data_integrity: bool) -> ValidationResult:
    """Validate /spines/meshes subgroups."""
    result = ValidationResult()

    group_names = list(meshes_grp.keys())
    if len(group_names) == 0:
        result.add_warning("/spines/meshes is empty (no mesh groups)")
        return result

    result.add_info(f"/spines/meshes contains {len(group_names)} group(s): {group_names}")

    for name in group_names:
        item = meshes_grp[name]
        if not isinstance(item, h5py.Group):
            result.add_error(f"/spines/meshes/{name} is not a group")
            continue

        keys = set(item.keys())

        # Required datasets within a mesh group
        if GRP_VERTICES not in keys:
            result.add_error(f"/spines/meshes/{name}: missing '{GRP_VERTICES}' dataset")
        if GRP_TRIANGLES not in keys:
            result.add_error(f"/spines/meshes/{name}: missing '{GRP_TRIANGLES}' dataset")
        if GRP_OFFSETS not in keys:
            result.add_error(f"/spines/meshes/{name}: missing '{GRP_OFFSETS}' dataset")

        # head_neck_values is optional
        if GRP_HEAD_NECK_VALUES in keys:
            result.add_info(f"/spines/meshes/{name}: head_neck_values present")

        if check_data_integrity:
            _validate_spine_mesh_data(item, name, result)

    return result


def _validate_mesh_datasets(
    grp: h5py.Group,
    path: str,
    result: ValidationResult,
    check_global_indices: bool = True,
) -> None:
    """Validate vertices and triangles datasets within a mesh group.

    Checks shape, NaN/Inf in vertices, negative triangle indices, and
    optionally triangle indices vs total vertex count.

    Args:
        grp: HDF5 group containing 'vertices' and 'triangles' datasets.
        path: Display path for error messages.
        result: ValidationResult to accumulate findings into.
        check_global_indices: If True, verify triangle indices < total vertex
            count. Set to False for spine meshes where indices are local
            per spine and validated separately via offsets.
    """
    keys = set(grp.keys())
    n_vertices = None

    if GRP_VERTICES in keys:
        vertices = grp[GRP_VERTICES]
        if isinstance(vertices, h5py.Dataset):
            if vertices.ndim != 2 or vertices.shape[1] != 3:
                result.add_error(f"{path}/vertices: expected shape (N, 3), got {vertices.shape}")
            else:
                n_vertices = vertices.shape[0]
                data = vertices[:]
                if np.any(np.isnan(data)):
                    result.add_error(f"{path}/vertices: contains NaN values")
                elif np.any(np.isinf(data)):
                    result.add_error(f"{path}/vertices: contains Inf values")

    if GRP_TRIANGLES in keys:
        triangles = grp[GRP_TRIANGLES]
        if isinstance(triangles, h5py.Dataset):
            if triangles.ndim != 2 or triangles.shape[1] != 3:
                result.add_error(f"{path}/triangles: expected shape (M, 3), got {triangles.shape}")
            else:
                tri_data = triangles[:]
                if np.any(tri_data < 0):
                    result.add_error(f"{path}/triangles: contains negative indices")
                elif (
                    check_global_indices
                    and n_vertices is not None
                    and np.any(tri_data >= n_vertices)
                ):
                    result.add_error(
                        f"{path}/triangles: contains indices >= vertex count ({n_vertices})"
                    )


def _validate_spine_mesh_data(mesh_grp: h5py.Group, name: str, result: ValidationResult) -> None:
    """Validate data integrity within a single /spines/meshes/{name} group."""
    keys = set(mesh_grp.keys())
    prefix = f"/spines/meshes/{name}"

    _validate_mesh_datasets(mesh_grp, prefix, result, check_global_indices=False)

    # Offsets: shape (num_spines+1, 2) or (num_spines+1, 3)
    if GRP_OFFSETS in keys:
        offsets = mesh_grp[GRP_OFFSETS]
        if isinstance(offsets, h5py.Dataset):
            if offsets.ndim != 2 or offsets.shape[1] not in (2, 3):
                result.add_error(
                    f"{prefix}/offsets: expected shape (K, 2) or (K, 3), got {offsets.shape}"
                )
            elif offsets.shape[0] < 2:
                result.add_error(
                    f"{prefix}/offsets: must have at least 2 rows (got {offsets.shape[0]})"
                )
            else:
                off_data = offsets[:]
                # First row should be zeros
                if not np.all(off_data[0, :2] == 0):
                    result.add_warning(f"{prefix}/offsets: first row is not (0, 0, ...)")
                # Offsets should be non-decreasing
                if np.any(np.diff(off_data[:, 0]) < 0):
                    result.add_error(f"{prefix}/offsets: vertex offsets are not non-decreasing")
                if np.any(np.diff(off_data[:, 1]) < 0):
                    result.add_error(f"{prefix}/offsets: triangle offsets are not non-decreasing")

                # Per-spine triangle index check: indices are local to each
                # spine's vertex slice, so they must be < (vertex_end - vertex_start)
                if GRP_TRIANGLES in keys and GRP_VERTICES in keys:
                    tri_data = mesh_grp[GRP_TRIANGLES][:]
                    n_spines = off_data.shape[0] - 1
                    for i in range(n_spines):
                        v_start, v_end = off_data[i, 0], off_data[i + 1, 0]
                        t_start, t_end = off_data[i, 1], off_data[i + 1, 1]
                        n_local_verts = v_end - v_start
                        if t_end > t_start and n_local_verts > 0:
                            spine_tris = tri_data[t_start:t_end]
                            if np.any(spine_tris >= n_local_verts):
                                result.add_error(
                                    f"{prefix}: spine {i} has triangle "
                                    f"indices >= local vertex count "
                                    f"({n_local_verts})"
                                )
                                break  # one error is enough


def _validate_soma_group(soma_grp: h5py.Group, check_data_integrity: bool) -> ValidationResult:
    """Validate the /soma group structure and data."""
    result = ValidationResult()

    keys = set(soma_grp.keys())

    if GRP_MESHES not in keys:
        result.add_warning(f"/soma: missing '{GRP_MESHES}' subgroup")
        return result

    meshes_grp = soma_grp[GRP_MESHES]
    if not isinstance(meshes_grp, h5py.Group):
        result.add_error(f"/soma/{GRP_MESHES} is not a group")
        return result

    neuron_names = list(meshes_grp.keys())
    if len(neuron_names) == 0:
        result.add_warning("/soma/meshes is empty (no soma meshes)")
        return result

    result.add_info(f"/soma/meshes contains {len(neuron_names)} mesh(es): {neuron_names}")

    for name in neuron_names:
        item = meshes_grp[name]
        if not isinstance(item, h5py.Group):
            result.add_error(f"/soma/meshes/{name} is not a group")
            continue

        keys_inner = set(item.keys())
        if GRP_VERTICES not in keys_inner:
            result.add_error(f"/soma/meshes/{name}: missing '{GRP_VERTICES}' dataset")
        if GRP_TRIANGLES not in keys_inner:
            result.add_error(f"/soma/meshes/{name}: missing '{GRP_TRIANGLES}' dataset")

        if check_data_integrity:
            _validate_mesh_datasets(item, f"/soma/meshes/{name}", result)

    return result


def _validate_cross_group_integrity(h5: h5py.File) -> ValidationResult:
    """Validate referential integrity across groups.

    Checks that:
    - Neuron names in /edges reference existing entries in /morphology.
    - spine_morphology values in /edges reference existing /spines/meshes subgroups
      (if meshes are present).
    - spine_id values in /edges are valid indices into the corresponding
      /spines/skeletons (root section count) and /spines/meshes (offset count).
    - afferent_section_id values in /edges are valid section indices in
      /morphology/{name}/structure.
    """
    result = ValidationResult()

    # Neuron names consistency
    morph_names = set(h5[GRP_MORPH].keys()) if GRP_MORPH in h5 else set()
    edges_names = set(h5[GRP_EDGES].keys()) if GRP_EDGES in h5 else set()
    mesh_names = (
        set(h5[f"{GRP_SPINES}/{GRP_MESHES}"].keys())
        if GRP_SPINES in h5 and GRP_MESHES in h5[GRP_SPINES]
        else set()
    )

    # /edges neuron names should exist in /morphology
    orphan_edges = edges_names - morph_names
    if orphan_edges:
        result.add_error(f"/edges references neuron(s) not in /morphology: {sorted(orphan_edges)}")

    # Check spine_morphology references in /edges point to valid meshes
    # (skeleton references are already checked per-edge in _validate_edges_data_integrity)
    if mesh_names:
        for edge_name in edges_names:
            edge_grp = h5[GRP_EDGES][edge_name]
            if not isinstance(edge_grp, h5py.Group):
                continue
            if "spine_morphology" not in edge_grp:
                continue

            spine_morph_dset = edge_grp["spine_morphology"]
            if not isinstance(spine_morph_dset, h5py.Dataset):
                continue

            spine_morph_values = set(spine_morph_dset[:].astype(str))

            missing_meshes = spine_morph_values - mesh_names
            if missing_meshes:
                result.add_warning(
                    f"/edges/{edge_name}/spine_morphology references mesh groups "
                    f"not in /spines/meshes: {sorted(missing_meshes)}"
                )

    # Check that spine_id values are valid indices into skeletons and meshes
    for edge_name in edges_names:
        edge_grp = h5[GRP_EDGES][edge_name]
        if not isinstance(edge_grp, h5py.Group):
            continue
        if "spine_morphology" not in edge_grp or "spine_id" not in edge_grp:
            continue

        spine_morph_dset = edge_grp["spine_morphology"]
        spine_id_dset = edge_grp["spine_id"]
        if not isinstance(spine_morph_dset, h5py.Dataset):
            continue
        if not isinstance(spine_id_dset, h5py.Dataset):
            continue

        morph_values = spine_morph_dset[:].astype(str)
        id_values = spine_id_dset[:]

        # Group spine_ids by their spine_morphology to check bounds per group
        unique_morphs = set(morph_values)
        for spine_morph in unique_morphs:
            mask = morph_values == spine_morph
            ids_for_morph = id_values[mask]
            max_id = int(np.max(ids_for_morph))

            # Check against skeleton root sections
            skel_path = f"{GRP_SPINES}/{GRP_SKELETONS}/{spine_morph}"
            if skel_path in h5:
                skel_grp = h5[skel_path]
                if "structure" in skel_grp:
                    structure = skel_grp["structure"]
                    if isinstance(structure, h5py.Dataset) and structure.ndim == 2:
                        # Root sections have parent == -1 (column index 2)
                        n_root_sections = int(np.sum(structure[:, 2] == -1))
                        if max_id >= n_root_sections:
                            result.add_error(
                                f"/edges/{edge_name}: spine_id={max_id} for "
                                f"spine_morphology='{spine_morph}' exceeds skeleton "
                                f"root section count ({n_root_sections})"
                            )

            # Check against mesh offsets
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{spine_morph}"
            if mesh_path in h5:
                mesh_grp = h5[mesh_path]
                if GRP_OFFSETS in mesh_grp:
                    offsets = mesh_grp[GRP_OFFSETS]
                    if isinstance(offsets, h5py.Dataset) and offsets.ndim == 2:
                        # Number of spines in mesh = offsets rows - 1
                        n_mesh_spines = offsets.shape[0] - 1
                        if max_id >= n_mesh_spines:
                            result.add_error(
                                f"/edges/{edge_name}: spine_id={max_id} for "
                                f"spine_morphology='{spine_morph}' exceeds mesh "
                                f"offset count ({n_mesh_spines})"
                            )

    # Check afferent_section_id references valid sections in /morphology
    for edge_name in edges_names:
        edge_grp = h5[GRP_EDGES][edge_name]
        if not isinstance(edge_grp, h5py.Group):
            continue
        if "afferent_section_id" not in edge_grp:
            continue

        morph_path = f"{GRP_MORPH}/{edge_name}"
        if morph_path not in h5:
            continue  # already reported as orphan edge

        morph_grp = h5[morph_path]
        if "structure" not in morph_grp:
            continue

        structure = morph_grp["structure"]
        if not isinstance(structure, h5py.Dataset) or structure.ndim != 2:
            continue

        n_sections = structure.shape[0]
        section_ids = edge_grp["afferent_section_id"][:]
        max_section_id = int(np.max(section_ids))
        if max_section_id >= n_sections:
            result.add_error(
                f"/edges/{edge_name}: afferent_section_id={max_section_id} "
                f"exceeds morphology section count ({n_sections})"
            )

    return result
