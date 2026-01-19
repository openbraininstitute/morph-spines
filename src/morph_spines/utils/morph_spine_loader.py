"""Loads the representation of a neron morphology with spines from file.

Provides reader functions to load the representation of a neuron morphology
with spines from an HDF5 file.
"""

import h5py
import morphio
import numpy as np
import pandas as pd
from morphio.mut import Morphology as MutableMorphology
from neurom.core.morphology import Morphology
from neurom.io.utils import load_morphology as neurom_load_morphology

from morph_spines.core.h5_schema import (
    ATT_VERSION,
    COL_SPINE_ID,
    COL_SPINE_MORPH,
    GRP_EDGES,
    GRP_METADATA,
    GRP_MORPH,
    GRP_SKELETONS,
    GRP_SPINES,
)
from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.core.soma import Soma
from morph_spines.core.spines import Spines


def _resolve_morphology_name(morphology_filepath: str, morphology_name: str | None = None) -> str:
    """Determine the morphology name from the given arguments.

    If morphology_name is None and a single morphology is found in the file, its name is returned;
    if a morphology name is given, it checks if it exists in the file, and if so, it is returned;
    a ValueError is raised otherwise.

    Args:
        morphology_filepath: HDF5 file path
        morphology_name: morphology name to load or None

    Returns: the morphology name to load data from
    """
    with h5py.File(morphology_filepath, "r") as h5:
        if GRP_MORPH in list(h5.keys()):
            lst_morph_names = list(h5[GRP_MORPH].keys())
            if len(lst_morph_names) == 0:
                raise ValueError("No morphology names were found in the file")
            if morphology_name is None:
                if len(lst_morph_names) > 1:
                    raise ValueError(
                        "Multiple morphology names found in the file: must specify a morphology "
                        "name"
                    )
                morphology_name = lst_morph_names[0]
            if morphology_name not in lst_morph_names:
                raise ValueError(f"Morphology {morphology_name} not found in file")
            return morphology_name
        else:
            raise ValueError("The file is not a valid morphology-with-spines file")


def load_morphology_with_spines(
    morphology_filepath: str,
    morphology_name: str | None = None,
    spines_are_centered: bool = True,
    process_subtrees: bool = False,
) -> MorphologyWithSpines:
    """Load a neuron morphology with spines.

    Loads a neuron morphology with spines from an hdf5 archive.
    Returns the representation of a spiny morphology of this package.
    """
    morphology = load_morphology(morphology_filepath, morphology_name, process_subtrees)
    soma = load_soma(morphology_filepath, morphology_name)
    spines = load_spines(morphology_filepath, morphology_name, spines_are_centered)
    return MorphologyWithSpines(morphology, soma, spines)


def load_morphology(
    filepath: str, name: str | None = None, process_subtrees: bool = False
) -> Morphology:
    """Load a neuron morphology from a neuron morphology with spines representation.

    Loads the basic neuron morphology without its spine representation.
    Returns the representation of the neuron morphology.
    """
    name = _resolve_morphology_name(filepath, name)
    coll = morphio.Collection(filepath)
    morphology = coll.load(f"{GRP_MORPH}/{name}")
    return Morphology(morphology, name, process_subtrees=process_subtrees)


def _is_pandas_dataframe_group(filepath: str, name: str) -> bool:
    """Check if an H5 group is a pandas dataframe."""
    with h5py.File(filepath, "r") as h5:
        if name not in h5:
            raise TypeError(f"Could not find {name} inside the H5 file")

        df_group = h5[name]
        if isinstance(df_group, h5py.Group):
            if "pandas_type" in df_group.attrs:
                return True
    return False


def _is_datasets_group(filepath: str, name: str) -> bool:
    """Check if an H5 group contains a set of datasets that form a table.

    The following conditions must be met:
    - 'name' must be a group inside the H5 file
    - 'name' group must contain at least one dataset
    - 'name' group cannot contain other groups, except for the 'metadata' group
    - All datasets within 'name' group must have the same size
    - All datasets within 'name' group cannot be multidimensional
    """
    with h5py.File(filepath, "r") as h5:
        if name not in h5:
            raise TypeError(f"Could not find {name} inside the H5 file")

        df_group = h5[name]

        # If 'name' is not a group, return false
        if not isinstance(df_group, h5py.Group):
            return False

        # If group is empty, return false
        if len(df_group.keys()) == 0:
            return False

        dsets_len = []
        for key, item in df_group.items():
            # Skip group metadata
            if key != GRP_METADATA:
                if not isinstance(item, h5py.Dataset):
                    return False

                # If dataset is a multidimensional array, return False
                if len(item.shape) > 1:
                    return False

                if item.shape == ():
                    # Scalar datasets
                    length = 0
                else:
                    # 1-dimensional arrays
                    length = item.shape[0]

                dsets_len.append(length)

        # All dataset lengths must be the same
        if len(set(dsets_len)) == 1:
            return True

    return False


def _load_spine_table_from_datasets_group(filepath: str, name: str) -> pd.DataFrame:
    """Load the spine table from a group of HDF5 datasets as a pandas dataframe.

    Note: all datasets with object ('O') or fixed-string ('S') are converted into strings.
    """
    columns = dict()
    with h5py.File(filepath, "r") as h5:
        df_group = h5[name]
        for key in df_group.keys():
            if key != GRP_METADATA:
                if df_group[key].shape == ():
                    # If it's a scalar type, create a 1-element array
                    col_data = np.array([df_group[key][()]])
                else:
                    col_data = df_group[key][:]
                # Convert byte strings into Python strings
                if col_data.dtype.kind == "S" or col_data.dtype.kind == "O":
                    col_data = col_data.astype(str)
                columns[str(key)] = col_data
    return pd.DataFrame(columns)


def _get_spine_table_version(filepath: str, name: str) -> tuple[int, int]:
    """Get the version of the spines table from metadata.

    Returns a tuple with major and minor version of the spines table.
    """
    major = 0
    minor = 1
    with h5py.File(filepath, "r") as h5:
        if not isinstance(h5[name], h5py.Group):
            return 0, 0
        if GRP_METADATA not in h5[name].keys():
            # If metadata group is not found, assume it's 0.1 version
            return major, minor

        metadata = h5[f"{name}/{GRP_METADATA}"]
        major, minor = metadata.attrs[ATT_VERSION]

        return major, minor


def load_spine_table(filepath: str, name: str) -> pd.DataFrame:
    """Load the spines table from a neuron morphology with spines representation.

    Returns the spines table as a pandas DataFrame.
    """
    major, minor = _get_spine_table_version(filepath, name)

    if major == 0 and minor == 1:
        # Pandas DataFrame format
        if not _is_pandas_dataframe_group(filepath, name):
            raise TypeError(f"Could not find a valid spine table in {name} for version 0.1")
        else:
            print(
                "Warning: deprecated format: spine table stored as pandas DataFrame in HDF5 file."
                "\nPlease, use the conversion script 'h5_dataframe_to_h5_datasets_group.py' to "
                "update the format."
            )
            spine_table = pd.read_hdf(filepath, key=name)
            spine_table = (
                spine_table.to_frame() if isinstance(spine_table, pd.Series) else spine_table
            )

    elif major == 1 and minor == 0:
        # Group of datasets format
        if not _is_datasets_group(filepath, name):
            raise TypeError(f"Could not find a valid spine table in {name} for version 1.0")
        else:
            spine_table = _load_spine_table_from_datasets_group(filepath, name)

    else:
        raise TypeError(
            f"Could not find a valid spine table in {name}. Unsupported version {major}.{minor}"
        )

    return spine_table


def load_spine_skeletons(filepath: str, name: str, spine_table: pd.DataFrame) -> Morphology:
    """Given a spine table, load all the spine skeletons present in the table.

    Args:
        filepath: H5 file containing the spine skeletons
        name: Neuron ID to which the spines belong to
        spine_table: Spine table in the form of a Pandas DataFrame

    Returns: a single NeuroM Morphology object containing all the spine skeletons
    """
    # Check whether all spines must be loaded from the same H5 dataset (I/O optimization)
    skeletons_table = spine_table[[COL_SPINE_ID, COL_SPINE_MORPH]]
    skeletons_datasets = set(skeletons_table[COL_SPINE_MORPH])
    skeletons_collections = {}
    coll = morphio.Collection(filepath)
    for collection in skeletons_datasets:
        skeletons_collections[collection] = neurom_load_morphology(
            coll.load(f"{GRP_SPINES}/{GRP_SKELETONS}/{collection}")
        )

    # Optimization: if spines are organized by morphology name, and we want to load its spines only
    if len(skeletons_datasets) == 1:
        morph_name = list(skeletons_datasets)[0]
        skeletons_indices = skeletons_table[COL_SPINE_ID]
        if len(set(skeletons_indices)) == len(skeletons_indices):
            if len(skeletons_collections[morph_name].neurites) == len(skeletons_indices):
                return skeletons_collections[morph_name]

    spine_skeletons = MutableMorphology()

    for row in skeletons_table.itertuples(index=False):
        spine_morph = getattr(row, COL_SPINE_MORPH)
        spine_id = getattr(row, COL_SPINE_ID)
        spine = skeletons_collections[spine_morph].to_morphio().root_sections[spine_id]

        spine_skeletons.append_root_section(spine, recursive=True)

    return neurom_load_morphology(spine_skeletons.as_immutable())


def load_spines(filepath: str, name: str | None = None, spines_are_centered: bool = True) -> Spines:
    """Load the spines from a neuron morphology with spines representation.

    Loads the spines of a 'neuron morphology with spines' from an HDF5 file.
    Returns the representation of the spines.
    """
    name = _resolve_morphology_name(filepath, name)

    spine_table_path = f"{GRP_EDGES}/{name}"
    spine_table = load_spine_table(filepath, spine_table_path)

    centered_spine_skeletons = load_spine_skeletons(filepath, name, spine_table)

    return Spines(
        filepath,
        name,
        spine_table,
        centered_spine_skeletons,
        spines_are_centered=spines_are_centered,
    )


def load_soma(filepath: str, name: str | None = None) -> Soma:
    """Load the soma mesh from a neuron morphology with spines representation."""
    name = _resolve_morphology_name(filepath, name)
    return Soma(filepath, name)
