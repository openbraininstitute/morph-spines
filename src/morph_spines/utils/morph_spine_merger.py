"""Merge multiple morphology-with-spines HDF5 files into a single output file.

Provides a public function to merge morph-spines files with optional renaming
of neuron keys and spines library names via a single rename map. The
spine_morphology entry in the spines tables is updated automatically to
reflect any renamed groups.
"""

import logging
import warnings
from pathlib import Path

import h5py
import numpy as np

from morph_spines.core.h5_schema import (
    COL_SPINE_MORPH,
    GRP_EDGES,
    GRP_MESHES,
    GRP_MORPH,
    GRP_SKELETONS,
    GRP_SOMA,
    GRP_SPINES,
)

L = logging.getLogger(__name__)


def merge_morphologies_with_spines(
    source_files: list[Path],
    output_path: Path,
    rename_map: dict[tuple[Path, str], str] | None = None,
    include_meshes: bool = True,
) -> None:
    """Merge multiple morph-with-spines HDF5 files into one with optional renaming.

    Copies all morphologies, spines tables, spines skeletons, and, optionally, soma and
    spines meshes from the source files into a single output file.

    Neuron keys and shared spines library names can be renamed via rename_map. When a neuron
    is renamed, its neuron-specific spines group (name == neuron name) is automatically
    renamed to match. Shared spines libraries (name != any neuron name in the file) can also
    be renamed via the same map. In this case, their corresponding entries in the spines
    table's spine_morphology entry will be updated accordingly.

    This function can also be used to rename the entries of a single morphology-with-spines
    file into the new output file (source files are never modified).

    Args:
        source_files: List of morph-with-spines HDF5 file paths to merge.
        output_path: Path for the output HDF5 file. File must not already exist.
        rename_map: Optional, mapping from (source_file_path, original_name) to desired
            destination name. The original_name can be either a morphology key or a shared
            spines library name. This allows duplicate names across source files as long as
            they map to unique destination names. Keys not in the map will be copied keeping
            their original name.
        include_meshes: If True (default), copy /soma/meshes and /spines/meshes groups.
            If False, omit them for a smaller output file.

    Raises:
        ValueError:
            - If source_files is empty.
            - If a source file does not comply to the morphology-with-spines file format.
            - If destination names (morph or spines group) collide in the output.
        FileExistsError: If output_path already exists.
    """
    if not source_files:
        raise ValueError("source_files must not be empty")

    # Normalize paths so str vs Path mismatches don't cause silent rename_map misses
    source_files = [Path(p) for p in source_files]
    output_path = Path(output_path)

    if output_path.exists():
        raise FileExistsError(f"Output file already exists: {output_path}")

    rename_map = {(Path(p), name): dest for (p, name), dest in (rename_map or {}).items()}

    _validate_sources(source_files, rename_map)

    # Pre-compute per-file name maps
    file_name_maps: dict[Path, dict[str, str]] = {}
    for (path, name), dest in rename_map.items():
        if path not in file_name_maps:
            file_name_maps[path] = {}
        file_name_maps[path][name] = dest

    L.info("Merging %d source file(s) into %s", len(source_files), output_path)

    with h5py.File(output_path, "w") as h5_out:
        for i, src_path in enumerate(source_files, 1):
            L.info("Processing file %d/%d: %s", i, len(source_files), src_path)
            with h5py.File(src_path, "r") as h5_in:
                _copy_source_file(h5_in, h5_out, file_name_maps.get(src_path, {}), include_meshes)

    L.info("Merge complete: %s", output_path)


def _validate_sources(
    source_files: list[Path],
    rename_map: dict[tuple[Path, str], str],
) -> None:
    """Validate source file format and check for destination name collisions."""
    from morph_spines.utils.morph_spine_validator import (
        validate_morph_with_spines_file,
    )

    morph_dest_names: list[str] = []
    spine_dest_names: list[str] = []
    valid_keys: set[tuple[Path, str]] = set()

    for src_path in source_files:
        # Use the file validator for structural checks
        result = validate_morph_with_spines_file(src_path)
        if not result.is_valid:
            raise ValueError(
                f"Invalid file {src_path}:\n" + "\n".join(f"  - {e}" for e in result.errors)
            )

        with h5py.File(src_path, "r") as h5:
            skeletons_grp = h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]

            # Collect valid keys and destination names for this file
            for k in h5[GRP_MORPH].keys():
                valid_keys.add((src_path, k))
                morph_dest_names.append(rename_map.get((src_path, k), k))
            for k in skeletons_grp.keys():
                valid_keys.add((src_path, k))
                spine_dest_names.append(rename_map.get((src_path, k), k))

    # Check uniqueness
    _check_duplicates(morph_dest_names, "morphology")
    _check_duplicates(spine_dest_names, "spines library")

    # Warn about unused rename_map entries
    unused = set(rename_map.keys()) - valid_keys
    if unused:
        warnings.warn(
            f"rename_map contains entries not found in source files: {sorted(unused)}",
            stacklevel=3,
        )


def _check_duplicates(names: list[str], label: str) -> None:
    """Raise ValueError if any name appears more than once."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(f"Duplicate {label} destination name: '{name}'")
        seen.add(name)


def _copy_source_file(
    h5_in: h5py.File,
    h5_out: h5py.File,
    name_map: dict[str, str],
    include_meshes: bool,
) -> None:
    """Copy all groups from one source file into the output."""
    # /morphology/{name}
    for name in h5_in[GRP_MORPH].keys():
        dst_grp = h5_out.require_group(GRP_MORPH)
        h5_in.copy(h5_in[f"{GRP_MORPH}/{name}"], dst_grp, name=name_map.get(name, name))

    # /edges/{name} -- spines table; spine_morphology column may need update
    for name in h5_in[GRP_EDGES].keys():
        _copy_spines_table(h5_in, h5_out, name, name_map.get(name, name), name_map)

    # /soma/meshes/{name} (optional)
    if include_meshes:
        soma_mesh_path = f"{GRP_SOMA}/{GRP_MESHES}"
        src_parent = h5_in.get(soma_mesh_path)
        if src_parent is not None:
            dst_grp = h5_out.require_group(soma_mesh_path)
            for name in src_parent.keys():
                h5_in.copy(src_parent[name], dst_grp, name=name_map.get(name, name))

    # /spines/skeletons/{name}
    skel_parent = h5_in[f"{GRP_SPINES}/{GRP_SKELETONS}"]
    dst_grp = h5_out.require_group(f"{GRP_SPINES}/{GRP_SKELETONS}")
    for name in skel_parent.keys():
        h5_in.copy(skel_parent[name], dst_grp, name=name_map.get(name, name))

    # /spines/meshes/{name} (optional)
    if include_meshes:
        mesh_parent = h5_in.get(f"{GRP_SPINES}/{GRP_MESHES}")
        if mesh_parent is not None:
            dst_grp = h5_out.require_group(f"{GRP_SPINES}/{GRP_MESHES}")
            for name in mesh_parent.keys():
                h5_in.copy(mesh_parent[name], dst_grp, name=name_map.get(name, name))


def _copy_spines_table(
    h5_in: h5py.File,
    h5_out: h5py.File,
    src_name: str,
    neuron_dest_name: str,
    name_map: dict[str, str],
) -> None:
    """Copy one spines table and update with optional renamings.

    Source spines table (/edges/{src}) is copied to output /edges/{dest}, updating
    the spine_morphology entry with the renamed values in name_map.
    """
    src_grp = h5_in[f"{GRP_EDGES}/{src_name}"]
    dst_grp = h5_out.require_group(GRP_EDGES).create_group(neuron_dest_name)

    # Determine which spine_morphology values need updating
    raw = src_grp[COL_SPINE_MORPH][:]
    spine_morph_values = np.array(
        [v.decode() if isinstance(v, bytes) else str(v) for v in raw], dtype=object
    )
    needs_update = False
    for old_name in set(spine_morph_values):
        new_name = name_map.get(old_name, old_name)
        if new_name != old_name:
            spine_morph_values[spine_morph_values == old_name] = new_name
            needs_update = True

    dt = h5py.string_dtype(encoding="utf-8")
    for ds_name in src_grp:
        if ds_name == COL_SPINE_MORPH and needs_update:
            # spine_morphology entry, only if it needs to be updated
            dst_grp.create_dataset(COL_SPINE_MORPH, data=spine_morph_values, dtype=dt)
        else:
            # The rest of the spines table is copied as-is, including metadata group
            h5_in.copy(src_grp[ds_name], dst_grp, name=ds_name)
