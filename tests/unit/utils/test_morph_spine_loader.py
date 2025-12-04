from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from morph_spines import Soma
from morph_spines.core.h5_schema import GRP_EDGES, GRP_MORPH
from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.utils.morph_spine_loader import (
    _is_compound_dataset,
    _is_datasets_group,
    _is_pandas_dataframe_group,
    _load_spine_table_from_compound_dataset,
    _resolve_morphology_name,
    load_morphology_with_spines,
    load_soma,
    load_spine_table,
)

SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_DATASET_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_datasets.h5"
SAMPLE_MORPH_WITH_SPINES_COMPOUND_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_compound.h5"
ALL_SAMPLE_FILES = [SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, SAMPLE_MORPH_WITH_SPINES_COMPOUND_FILE]
MORPH_WITH_SPINES_ID = "01234"


def test__resolve_morphology_name_single():
    for test_file in ALL_SAMPLE_FILES:
        name = _resolve_morphology_name(test_file)
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
        _resolve_morphology_name(SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, "m1")


def test__resolve_morphology_name_empty_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group(GRP_MORPH)

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f))


def test__resolve_morphology_name_invalid_file(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("invalid_group")

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f))


def test__is_pandas_dataframe_group_true(tmp_path):
    f = tmp_path / "test.h5"
    df = pd.DataFrame([[1, 2], [3, 4]])
    df.to_hdf(f, key="df", mode="w")

    assert _is_pandas_dataframe_group(str(f), "df")


def test__is_pandas_dataframe_group_invalid(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("root_group")

    with pytest.raises(TypeError):
        _is_pandas_dataframe_group(str(f), "invalid_group")


def test__is_pandas_dataframe_group_false():
    assert not _is_pandas_dataframe_group(SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, GRP_EDGES)
    assert not _is_pandas_dataframe_group(SAMPLE_MORPH_WITH_SPINES_COMPOUND_FILE, GRP_EDGES)


def test__is_datasets_group_true_1dim_datasets():
    assert _is_datasets_group(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )


def test__is_datasets_group_true_scalar_datasets(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("scalar_int", data=2)
        root_grp.create_dataset("scalar_float", data=1.23)

    assert _is_datasets_group(str(f), "root_group")


def test__is_datasets_group_false_no_group():
    assert not _is_datasets_group(
        SAMPLE_MORPH_WITH_SPINES_COMPOUND_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )


def test__is_datasets_group_false_empty_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("empty_group")
    assert not _is_datasets_group(str(f), "empty_group")


def test__is_datasets_group_false_nested_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("root_group", data=np.array([1, 2, 3, 4]))
        root_grp.create_group("nested_group")
    assert not _is_datasets_group(str(f), "root_group")


def test__is_datasets_group_false_multiple_len_datasets(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("len4", data=np.array([1, 2, 3, 4]))
        root_grp.create_dataset("len2", data=np.array([5, 6]))
    assert not _is_datasets_group(str(f), "root_group")


def test__is_datasets_group_false_ndim_datasets(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("root_group", data=np.array([[1, 2], [3, 4]]))
    assert not _is_datasets_group(str(f), "root_group")


def test__is_datasets_group_invalid(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("root_group")

    with pytest.raises(TypeError):
        _is_datasets_group(str(f), "invalid_group")


def test__is_compound_datasets_true():
    assert _is_compound_dataset(
        SAMPLE_MORPH_WITH_SPINES_COMPOUND_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )


def test__is_compound_dataset_false_no_dataset():
    assert not _is_compound_dataset(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )


def test__is_compound_dataset_invalid(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("root_group")

    with pytest.raises(TypeError):
        _is_compound_dataset(str(f), "invalid_group")


def test_load_spine_table_success():
    for test_file in ALL_SAMPLE_FILES:
        df = load_spine_table(test_file, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}")

        assert isinstance(df, pd.DataFrame)
        assert len(df.columns) == 20
        assert set(
            [
                "afferent_surface_x",
                "afferent_center_x",
                "spine_length",
                "spine_orientation_vector_x",
                "spine_rotation_x",
                "afferent_section_id",
            ]
        ).issubset(set(df.columns))
        assert df.loc[0, "afferent_surface_x"] == np.float64(0.1)
        assert df.loc[1, "spine_length"] == 2


def test_load_spine_table_pandas_df(tmp_path):
    f = tmp_path / "test.h5"
    test_df = pd.DataFrame([[1, 2], [3, 4]])
    test_df.to_hdf(f, key="df", mode="w")
    df = load_spine_table(str(f), "df")

    assert test_df.equals(df)


def test_load_spine_table_scalar_datasets(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("scalar_int", data=2)
        root_grp.create_dataset("scalar_float", data=1.23)
    df = load_spine_table(str(f), "root_group")

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(
        [
            "scalar_int",
            "scalar_float",
        ]
    ) == set(df.columns)
    assert df.loc[0, "scalar_int"] == 2
    assert df.loc[0, "scalar_float"] == 1.23


def test_load_spine_table_invalid(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("np_array", data=np.array([[1, 2], [3, 4]]))

    with pytest.raises(TypeError):
        load_spine_table(str(f), f"{GRP_EDGES}/np_array")


def test__load_spine_table_from_compound_dataset_not_compound(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("list", data=np.array([1, 2, 3, 4]))

    with pytest.raises(TypeError):
        _load_spine_table_from_compound_dataset(str(f), f"{GRP_EDGES}/list")


def test__load_spine_table_from_compound_dataset_obj_dtype(tmp_path):
    dt = np.dtype([("objects", h5py.string_dtype(encoding="utf-8"))])
    data = np.zeros(2, dtype=dt)
    data["objects"] = ["obj1", "obj2"]
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("spines", data=data)

    df = _load_spine_table_from_compound_dataset(str(f), f"{GRP_EDGES}/spines")

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 1
    assert set(["objects"]) == set(df.columns)
    assert str(df.loc[0, "objects"]) == str("obj1")
    assert str(df.loc[1, "objects"]) == str("obj2")


def test_load_soma(tmp_path):
    soma = load_soma(SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, MORPH_WITH_SPINES_ID)

    assert isinstance(soma, Soma)


def test_load_morphology_with_spines():
    for test_file in ALL_SAMPLE_FILES:
        morph_with_spines = load_morphology_with_spines(test_file, spines_are_centered=False)

        assert isinstance(morph_with_spines, MorphologyWithSpines)
