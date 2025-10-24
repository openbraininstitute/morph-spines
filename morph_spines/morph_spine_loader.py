import h5py
import pandas

import morphio
from neurom.io.utils import load_morphology

from .morphology_with_spines import (
    GRP_EDGES,
    GRP_MORPH,
    GRP_SPINES,
    GRP_SKELETONS,
    MorphologyWithSpines,
)

def load_morphology_with_spines(
    morphology_fn, morphology_name=None, spines_are_centered=True, process_subtrees=False
):
    """Load a neuron morphology with spines.
    Loads a neuron morphology with spines from a hdf5 archive.
    Returns the representation of a spiny morphology of this package.
    """
    with h5py.File(morphology_fn, "r") as h5:
        lst_morph_names = list(h5[GRP_EDGES].keys())
        if len(lst_morph_names) == 0:
            raise ValueError("The file is not a valid morphology-with-spines file!")
        if morphology_name is None:
            if len(lst_morph_names) > 1:
                raise ValueError("Must specify morphology name!")
            morphology_name = lst_morph_names[0]
        if morphology_name not in lst_morph_names:
            raise ValueError(f"Morphology {morphology_name} not found in file!")
    spine_table = pandas.read_hdf(morphology_fn, key=GRP_EDGES + "/" + morphology_name)
    coll = morphio.Collection(morphology_fn)
    centered_spine_skeletons = load_morphology(
        coll.load(GRP_SPINES + "/" + GRP_SKELETONS + "/" + morphology_name)
    )
    smooth_morphology = coll.load(GRP_MORPH + "/" + morphology_name)
    return MorphologyWithSpines(
        morphology_fn,
        morphology_name,
        smooth_morphology,
        spine_table,
        centered_spine_skeletons,
        spines_are_centered=spines_are_centered,
        process_subtrees=process_subtrees,
    )