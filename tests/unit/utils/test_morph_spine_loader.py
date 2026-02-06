import h5py
import numpy as np
import pandas as pd
import pytest

from morph_spines.core.h5_schema import GRP_EDGES, GRP_MORPH
from morph_spines.utils.morph_spine_loader import (
    _is_datasets_group,
    _is_pandas_dataframe_group,
    _resolve_morphology_name,
    load_spine_table,
)


def test__resolve_morphology_name_single(tmp_path):
    f = tmp_path / "test.h5"
    morph_name = "m1"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_MORPH)
        grp.create_group(morph_name)

    name = _resolve_morphology_name(str(f))
    assert name == morph_name


def test__resolve_morphology_name_multiple_without_arg(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_MORPH)
        grp.create_group("m1")
        grp.create_group("m2")

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f))


def test__resolve_morphology_name_not_found(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_MORPH)
        grp.create_group("m1")

    with pytest.raises(ValueError):
        _resolve_morphology_name(str(f), "m2")


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


def test__is_pandas_dataframe_group_true_with_version(tmp_path):
    f = tmp_path / "test.h5"
    df = pd.DataFrame([[1, 2], [3, 4]])
    df.to_hdf(f, key="df", mode="w")

    with h5py.File(f, "a") as h5:
        root_grp = h5["df"]
        metadata_grp = root_grp.create_group("metadata")
        metadata_grp.attrs["version"] = np.array([0, 1], dtype=np.uint32)

    assert _is_pandas_dataframe_group(str(f), "df")


def test__is_pandas_dataframe_group_invalid(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        h5.create_group("root_group")

    with pytest.raises(TypeError):
        _is_pandas_dataframe_group(str(f), "invalid_group")


def test__is_pandas_dataframe_group_false(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        morph_grp = grp.create_group("m1")
        morph_grp.create_dataset("dataset", data=np.array([1, 2]))

    assert not _is_pandas_dataframe_group(str(f), GRP_EDGES)


def test__is_pandas_dataframe_group_false_no_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group("root_group")
        grp.create_dataset("dataset", data=np.array([1, 2, 3, 4]))

    assert not _is_pandas_dataframe_group(str(f), "root_group/dataset")


def test__is_datasets_group_true_1dim_datasets(tmp_path):
    f = tmp_path / "test.h5"
    morph_name = "m1"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        morph_grp = grp.create_group(morph_name)
        morph_grp.create_dataset("dataset_1", data=np.array([1, 2]))
        morph_grp.create_dataset("dataset_2", data=np.array([3, 4]))
        morph_grp.create_dataset("dataset_3", data=np.array([5, 6]))

    assert _is_datasets_group(str(f), f"{GRP_EDGES}/{morph_name}")


def test__is_datasets_group_true_scalar_datasets(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("scalar_int", data=2)
        root_grp.create_dataset("scalar_float", data=1.23)

    assert _is_datasets_group(str(f), "root_group")


def test__is_datasets_group_false_no_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        root_grp = h5.create_group("root_group")
        root_grp.create_dataset("dataset", data=np.array([1, 2]))

    assert not _is_datasets_group(str(f), "root_group/dataset")


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


def test_load_spine_table_success(tmp_path):
    df_ref = pd.DataFrame([[1, 2], [3, 4], [5, 6]], columns=["a", "b"])
    f = tmp_path / "test.h5"
    df_ref.to_hdf(str(f), key="/df_group")

    df = load_spine_table(str(f), "df_group")

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == len(df_ref.columns)
    assert set(df.columns) == set(df_ref.columns)
    assert df.loc[0, "a"] == df_ref.loc[0, "a"]
    assert df.loc[1, "b"] == df_ref.loc[1, "b"]


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
        metadata_grp = root_grp.create_group("metadata")
        metadata_grp.attrs["version"] = np.array([1, 0], dtype=np.uint32)
    df = load_spine_table(str(f), "root_group")

    assert isinstance(df, pd.DataFrame)
    assert len(df.columns) == 2
    assert set(["scalar_int", "scalar_float"]) == set(df.columns)
    assert df.loc[0, "scalar_int"] == 2
    assert df.loc[0, "scalar_float"] == 1.23


def test_load_spine_table_invalid(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("np_array", data=np.array([[1, 2], [3, 4]]))

    with pytest.raises(TypeError):
        load_spine_table(str(f), f"{GRP_EDGES}/np_array")


def test_load_spine_table_invalid_with_version01(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("np_array", data=np.array([[1, 2], [3, 4]]))
        metadata_grp = grp.create_group("metadata")
        metadata_grp.attrs["version"] = np.array([0, 1], dtype=np.uint32)

    with pytest.raises(TypeError):
        load_spine_table(str(f), f"{GRP_EDGES}")


def test_load_spine_table_invalid_with_version10(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group(GRP_EDGES)
        grp.create_dataset("np_array", data=np.array([[1, 2], [3, 4]]))
        metadata_grp = grp.create_group("metadata")
        metadata_grp.attrs["version"] = np.array([1, 0], dtype=np.uint32)

    with pytest.raises(TypeError):
        load_spine_table(str(f), f"{GRP_EDGES}")
