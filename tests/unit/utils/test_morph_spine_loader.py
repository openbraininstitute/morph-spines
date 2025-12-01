from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trimesh
from numpy.ma.testutils import assert_array_equal

from morph_spines import Soma
from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.core.spines import Spines

SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema.h5"
MORPH_WITH_SPINES_ID = "01234"

import h5py
import numpy as np
import pandas as pd
import pytest

from unittest.mock import MagicMock, patch

from morph_spines.core.h5_schema import GRP_EDGES, GRP_MORPH, GRP_SOMA, GRP_SKELETONS, GRP_SPINES
from morph_spines.utils.morph_spine_loader import (
    _resolve_morphology_name,
    _is_pandas_dataframe_group,
    load_spine_table,
    load_spines,
    load_morphology_with_spines,
    load_soma,
)


def test__resolve_morphology_name_single():
    name = _resolve_morphology_name(SAMPLE_MORPH_WITH_SPINES_FILE)
    assert name == MORPH_WITH_SPINES_ID


def test__resolve_morphology_name_multiple_without_arg(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_MORPH)
        grp.create_group("m1")
        grp.create_group("m2")

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f))


def test__resolve_morphology_name_not_found():
    with pytest.raises(ValueError):
        _resolve_morphology_name(SAMPLE_MORPH_WITH_SPINES_FILE, "m1")

def test__resolve_morphology_name_empty_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_MORPH)

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f))


def test__resolve_morphology_name_invalid_file(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group("invalid_group")

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f))


def test__is_pandas_dataframe_group_true(tmp_path):
    f = tmp_path / "test.h5"
    df = pd.DataFrame([[1, 2], [3, 4]])
    df.to_hdf(f, key="df", mode="w")

    assert _is_pandas_dataframe_group(str(f), "df")


def test__is_pandas_dataframe_group_false():
    assert not _is_pandas_dataframe_group(SAMPLE_MORPH_WITH_SPINES_FILE, GRP_EDGES)


def test_load_spine_table_array(tmp_path):
    df = load_spine_table(SAMPLE_MORPH_WITH_SPINES_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}")

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 20
    assert set([
        "afferent_surface_x",
        "afferent_center_x",
        "spine_length",
        "spine_orientation_vector_x",
        "spine_rotation_x",
        "afferent_section_id",
    ]).issubset(set(df.columns))
    assert df.loc[0, "afferent_surface_x"] == np.float64(0.1)
    assert df.loc[1, "spine_length"] == 2


def test_load_spine_table_pandas_df(tmp_path):
    f = tmp_path / "test.h5"

    test_df = pd.DataFrame([[1, 2], [3, 4]])
    test_df.to_hdf(f, key="df", mode="w")

    df = load_spine_table(str(f), "df")
    assert test_df.equals(df)


def test_load_spine_table_invalid(tmp_path):
    f = tmp_path / "test.h5"

    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("np_array", data=np.array([[1, 2],[3, 4]]))

    with pytest.raises(TypeError):
        load_spine_table(str(f), f"{GRP_EDGES}/np_array")


def test_load_morphology_with_spines():
    morph_with_spines = load_morphology_with_spines(
        SAMPLE_MORPH_WITH_SPINES_FILE,
        spines_are_centered=False
    )
    assert isinstance(morph_with_spines, MorphologyWithSpines)


def test_load_soma(tmp_path):
    soma = load_soma(SAMPLE_MORPH_WITH_SPINES_FILE, MORPH_WITH_SPINES_ID)
    assert isinstance(soma, Soma)

