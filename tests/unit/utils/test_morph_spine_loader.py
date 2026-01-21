from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from morph_spines import Soma
from morph_spines.core.h5_schema import GRP_EDGES, GRP_MORPH, GRP_SKELETONS, GRP_SPINES
from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.utils.morph_spine_loader import (
    _is_datasets_group,
    _is_pandas_dataframe_group,
    _resolve_morphology_name,
    load_morphology_with_spines,
    load_soma,
    load_spine_skeletons,
    load_spine_table,
)

SAMPLE_DATA_DIR = f"{Path(__file__).parent.parent}/data"
SAMPLE_MORPH_WITH_SPINES_DATASET_FILE = f"{SAMPLE_DATA_DIR}/morph_with_spines_schema_v1.0.h5"
MORPH_WITH_SPINES_ID = "neuron_0"

SAMPLE_MORPH_WITH_SPINES_COLLECTION_FILE = (
    f"{SAMPLE_DATA_DIR}/morph_with_spines_col_v1.0_2nrn_3col.h5"
)
COL_MORPH_IDS = ["neuron_0", "neuron_1"]
COL_COLLECTION_IDS = ["coll_0", "coll_1", "coll_2"]


def test__resolve_morphology_name_single():
    name = _resolve_morphology_name(SAMPLE_MORPH_WITH_SPINES_DATASET_FILE)
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


def test__is_pandas_dataframe_group_false():
    assert not _is_pandas_dataframe_group(SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, GRP_EDGES)


def test__is_pandas_dataframe_group_false_no_group(tmp_path):
    f = tmp_path / "test.h5"
    with h5py.File(f, "w") as h5:
        grp = h5.create_group("root_group")
        grp.create_dataset("dataset", data=np.array([1, 2, 3, 4]))

    assert not _is_pandas_dataframe_group(str(f), "root_group/dataset")


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
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE,
        f"{GRP_SPINES}/{GRP_SKELETONS}/{MORPH_WITH_SPINES_ID}/points",
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


def test_load_spine_table_success():
    df = load_spine_table(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )

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
    assert df.loc[0, "afferent_surface_x"] == np.float64(2.0)
    assert df.loc[1, "spine_length"] == np.float64(3.0)


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


def test_load_spine_skeletons_edgecase_spine_dup():
    df = load_spine_table(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )
    df.drop(index=0, inplace=True)
    df = pd.concat([df, df], axis=0, ignore_index=True)
    spines_skeletons = load_spine_skeletons(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, MORPH_WITH_SPINES_ID, df
    )
    num_spines = df.shape[0]

    assert num_spines == len(spines_skeletons.neurites)


def test_load_spine_skeletons_edgecase_spine_missing():
    df = load_spine_table(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, f"{GRP_EDGES}/{MORPH_WITH_SPINES_ID}"
    )
    df.drop(index=0, inplace=True)
    spines_skeletons = load_spine_skeletons(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, MORPH_WITH_SPINES_ID, df
    )
    num_spines = df.shape[0]

    assert num_spines == len(spines_skeletons.neurites)


def test_load_soma(tmp_path):
    soma = load_soma(SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, MORPH_WITH_SPINES_ID)

    assert isinstance(soma, Soma)


def test_load_morphology_with_spines_full_morph():
    morph_with_spines = load_morphology_with_spines(
        SAMPLE_MORPH_WITH_SPINES_DATASET_FILE, spines_are_centered=False
    )

    assert isinstance(morph_with_spines, MorphologyWithSpines)


def test_load_morphology_with_spines_from_collection():
    morph_with_spines = load_morphology_with_spines(
        SAMPLE_MORPH_WITH_SPINES_COLLECTION_FILE, morphology_name=COL_MORPH_IDS[0]
    )
    num_spines = 2 * len(COL_COLLECTION_IDS)

    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert num_spines == morph_with_spines.spines.spine_count
