import pandas as pd

from morph_spines.core.h5_schema import GRP_EDGES
from morph_spines.utils.morph_spine_loader import (
    load_spine_skeletons,
    load_spine_table,
)


def test_load_spine_skeletons_edgecase_spine_dup(single_morph_spines_file, single_morph_id):
    df = load_spine_table(single_morph_spines_file, f"{GRP_EDGES}/{single_morph_id}")
    df.drop(index=0, inplace=True)
    df = pd.concat([df, df], axis=0, ignore_index=True)
    spines_skeletons = load_spine_skeletons(single_morph_spines_file, single_morph_id, df)
    num_spines = len(df)

    assert num_spines == len(spines_skeletons.neurites)


def test_load_spine_skeletons_edgecase_spine_missing(single_morph_spines_file, single_morph_id):
    df = load_spine_table(single_morph_spines_file, f"{GRP_EDGES}/{single_morph_id}")
    df.drop(index=0, inplace=True)
    spines_skeletons = load_spine_skeletons(single_morph_spines_file, single_morph_id, df)
    num_spines = len(df)

    assert num_spines == len(spines_skeletons.neurites)
