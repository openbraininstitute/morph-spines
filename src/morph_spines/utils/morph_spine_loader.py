import h5py
import pandas

import morphio
from neurom.core.morphology import Morphology
from neurom.io.utils import load_morphology as neurom_load_morphology

from morph_spines.core.h5_schema import GRP_EDGES, GRP_MORPH, GRP_SPINES, GRP_SKELETONS
from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.core.soma import Soma
from morph_spines.core.spines import Spines


def _resolve_morphology_name(
        morphology_filepath, morphology_name
):
    with h5py.File(morphology_filepath, "r") as h5:
        lst_morph_names = list(h5[GRP_MORPH].keys())
        if len(lst_morph_names) == 0:
            raise ValueError("The file is not a valid morphology-with-spines file")
        if morphology_name is None:
            if len(lst_morph_names) > 1:
                raise ValueError("Multiple morphology names found in the file: must specify a morphology name")
            morphology_name = lst_morph_names[0]
        if morphology_name not in lst_morph_names:
            raise ValueError(f"Morphology {morphology_name} not found in file")
    return morphology_name

def load_morphology_with_spines(
    morphology_filepath, morphology_name=None, spines_are_centered=True, process_subtrees=False
):
    """Load a neuron morphology with spines.
    Loads a neuron morphology with spines from a hdf5 archive.
    Returns the representation of a spiny morphology of this package.
    """
    morphology = load_morphology(morphology_filepath, morphology_name, process_subtrees)
    soma = load_soma(morphology_filepath, morphology_name)
    spines = load_spines(morphology_filepath, morphology_name, spines_are_centered)
    return MorphologyWithSpines(morphology, soma, spines)

def load_morphology(filepath, name=None, process_subtrees=False):
    """Load a neuron morphology from a neuron morphology with spines representation.
    Loads the basic neuron morphology without its spine representation.
    Returns the representation of the neuron morphology.
    """
    name = _resolve_morphology_name(filepath, name)
    coll = morphio.Collection(filepath)
    morphology = coll.load(f"{GRP_MORPH}/{name}")
    return Morphology(morphology, name, process_subtrees=process_subtrees)

def load_spines(filepath, name=None, spines_are_centered=True):
    """Load the spines from a neuron morphology with spines representation.
    Loads the spines of a 'neuron morphology with spines' from a hdf5 archive.
    Returns the representation of the spines.
    """
    name = _resolve_morphology_name(filepath, name)
    spine_table = pandas.read_hdf(filepath, key=f"{GRP_EDGES}/{name}")
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

def load_soma(filepath, name=None):
    """
    Load the soma mesh from a neuron morphology with spines representation.
    """
    name = _resolve_morphology_name(filepath, name)
    return Soma(
        filepath,
        name
    )
