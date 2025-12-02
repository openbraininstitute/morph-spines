"""Loads the representation of a neron morphology with spines from file.

Provides reader functions to load the representation of a neuron morphology
with spines from an HDF5 file.
"""

import h5py
import morphio
import numpy as np
import pandas as pd
from neurom.core.morphology import Morphology
from neurom.io.utils import load_morphology as neurom_load_morphology

from morph_spines.core.h5_schema import GRP_EDGES, GRP_MORPH, GRP_SKELETONS, GRP_SPINES
from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.core.soma import Soma
from morph_spines.core.spines import Spines


def _resolve_morphology_name(morphology_filepath: str, morphology_name: str | None = None) -> str:
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

    Loads a neuron morphology with spines from a hdf5 archive.
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


def _is_pandas_dataframe_group(filepath: str, name: str | None = None) -> bool:
    """Check if an H5 group is a pandas dataframe."""
    with h5py.File(filepath, "r") as h5:
        df_group = h5[name]
        if isinstance(df_group, h5py.Group):
            if "pandas_type" in df_group.attrs:
                return True
    return False


def _load_spine_table_from_array(filepath: str, name: str) -> pd.DataFrame:
    """Load the spine table from an HDF5 compound type array as a pandas dataframe."""
    with h5py.File(filepath, "r") as h5:
        dset = h5[name]

        # Must be a compound dataset
        if not isinstance(dset.dtype, np.dtype) or dset.dtype.names is None:
            raise TypeError(f"Dataset {name} is not a compound dataset")

        data = dset[:]  # structured numpy array
        columns = {}
        for name in dset.dtype.names:
            col = data[name]
            # Handle variable-length UTF-8 strings (dtype = object)
            if col.dtype.kind == "O":
                col = col.astype(str)
            # Handle fixed-length ASCII/UTF-8 strings (dtype = 'Sxx')
            elif col.dtype.kind == "S":
                col = col.astype(str)
            # else: No conversion needed for numeric types

            columns[name] = col

        return pd.DataFrame(columns)


def load_spine_table(filepath: str, name: str) -> pd.DataFrame:
    """Load the spines table from a neuron morphology with spines representation.

    Returns the spines table as a pandas DataFrame.
    """
    if _is_pandas_dataframe_group(filepath, name):
        print(
            "Warning: deprecated format: spine table stored as pandas DataFrame in HDF5 file.\n"
            "Please, use the conversion script 'h5_dataframe_to_h5_struct_array.py' to update "
            "the format."
        )
        spine_table = pd.read_hdf(filepath, key=name)
    else:
        spine_table = _load_spine_table_from_array(filepath, name)

    if not isinstance(spine_table, pd.DataFrame):
        raise TypeError(f"Error reading the spine table from {name}")

    return spine_table


def load_spines(filepath: str, name: str | None = None, spines_are_centered: bool = True) -> Spines:
    """Load the spines from a neuron morphology with spines representation.

    Loads the spines of a 'neuron morphology with spines' from a hdf5 archive.
    Returns the representation of the spines.
    """
    name = _resolve_morphology_name(filepath, name)

    spine_table_path = f"{GRP_EDGES}/{name}"
    spine_table = load_spine_table(filepath, spine_table_path)

    coll = morphio.Collection(filepath)
    centered_spine_skeletons = neurom_load_morphology(
        coll.load(f"{GRP_SPINES}/{GRP_SKELETONS}/{name}")
    )
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
