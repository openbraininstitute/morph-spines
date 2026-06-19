"""Unit tests for morph_spines.utils.morph_spine_validator."""

import h5py
import numpy as np
import pytest

from morph_spines.core.h5_schema import (
    ATT_VERSION,
    GRP_EDGES,
    GRP_MESHES,
    GRP_METADATA,
    GRP_MORPH,
    GRP_OFFSETS,
    GRP_SKELETONS,
    GRP_SPINES,
    GRP_TRIANGLES,
    GRP_VERTICES,
)
from morph_spines.utils.morph_spine_validator import (
    ValidationResult,
    validate_morph_with_spines_file,
)
from tests.conftest import (
    NEURON_NAME,
    SPINE_MORPH_NAME,
    write_minimal_valid_file,
)


class TestFileLevelChecks:
    """Tests for basic file-level validation checks."""

    def test_file_not_found(self, tmp_path):
        r = validate_morph_with_spines_file(tmp_path / "nonexistent.h5")
        assert not r.is_valid
        assert any("not found" in e for e in r.errors)

    def test_not_hdf5(self, tmp_path):
        p = tmp_path / "bad.h5"
        p.write_text("not an hdf5 file")
        r = validate_morph_with_spines_file(p)
        assert not r.is_valid
        assert any("Cannot open" in e for e in r.errors)

    def test_valid_file_structure_only(self, tmp_path):
        r = validate_morph_with_spines_file(write_minimal_valid_file(tmp_path / "test.h5"))
        assert r.is_valid
        assert not r.data_integrity_checked

    def test_valid_file_with_data_integrity(self, tmp_path):
        r = validate_morph_with_spines_file(
            write_minimal_valid_file(tmp_path / "test.h5"), check_data_integrity=True
        )
        assert r.is_valid
        assert r.data_integrity_checked

    def test_path_is_directory(self, tmp_path):
        r = validate_morph_with_spines_file(tmp_path)
        assert not r.is_valid
        assert any("not a file" in e for e in r.errors)

    def test_unexpected_top_level_groups(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            h5.create_group("unexpected_group")
        r = validate_morph_with_spines_file(filepath)
        assert r.is_valid
        assert any("unexpected_group" in w for w in r.warnings)

    @pytest.mark.parametrize(
        "groups,keyword",
        [
            ([GRP_EDGES, GRP_SPINES], "/morphology"),
            ([GRP_MORPH, GRP_SPINES], "/edges"),
            ([GRP_MORPH, GRP_EDGES], "/spines"),
        ],
        ids=["missing-morphology", "missing-edges", "missing-spines"],
    )
    def test_missing_required_group(self, tmp_path, groups, keyword):
        filepath = tmp_path / "test.h5"
        with h5py.File(filepath, "w") as h5:
            for g in groups:
                h5.create_group(g)
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any(keyword in e for e in r.errors)


class TestMorphologyGroup:
    """Tests for /morphology group validation."""

    @pytest.mark.parametrize("dataset", ["points", "structure"])
    def test_missing_dataset(self, tmp_path, dataset):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_MORPH}/{NEURON_NAME}/{dataset}"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any(f"missing '{dataset}'" in e for e in r.errors)

    def test_wrong_points_shape(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            grp.create_dataset("points", data=np.ones((3, 3), dtype=np.float32))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("shape (N, 4)" in e for e in r.errors)

    def test_points_not_float32_warning(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            grp.create_dataset("points", data=np.ones((3, 4), dtype=np.float64))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert r.is_valid
        assert any("float32" in w for w in r.warnings)

    @pytest.mark.parametrize(
        "bad_value,keyword",
        [
            (np.nan, "NaN"),
            (np.inf, "Inf"),
        ],
        ids=["nan", "inf"],
    )
    def test_points_bad_values(self, tmp_path, bad_value, keyword):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            data = np.ones((3, 4), dtype=np.float32)
            data[1, 2] = bad_value
            grp.create_dataset("points", data=data)
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any(keyword in e for e in r.errors)

    @pytest.mark.parametrize(
        "dataset,shape,dtype",
        [
            ("points", (0, 4), np.float32),
            ("structure", (0, 3), np.int32),
        ],
        ids=["empty-points", "empty-structure"],
    )
    def test_empty_dataset(self, tmp_path, dataset, shape, dtype):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp[dataset]
            grp.create_dataset(dataset, data=np.empty(shape, dtype=dtype))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("empty" in e for e in r.errors)

    def test_empty_morphology_group(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_MORPH}/{NEURON_NAME}"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("empty" in e for e in r.errors)

    def test_structure_wrong_shape(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["structure"]
            grp.create_dataset("structure", data=np.ones((3, 2), dtype=np.int32))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("shape (M, 3)" in e for e in r.errors)

    def test_morphology_entry_is_dataset_not_group(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            h5[GRP_MORPH].create_dataset(NEURON_NAME, data=np.zeros(3))
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("not a group" in e for e in r.errors)


class TestEdgesGroup:
    """Tests for /edges group validation."""

    def test_missing_mandatory_column(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_EDGES}/{NEURON_NAME}/spine_length"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("spine_length" in e for e in r.errors)

    def test_unknown_column_warning(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            h5[f"{GRP_EDGES}/{NEURON_NAME}"].create_dataset("unknown_col", data=np.zeros(3))
        r = validate_morph_with_spines_file(filepath)
        assert r.is_valid
        assert any("unknown_col" in w for w in r.warnings)

    def test_wrong_metadata_version(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            meta = h5[f"{GRP_EDGES}/{NEURON_NAME}/{GRP_METADATA}"]
            meta.attrs[ATT_VERSION] = np.array([99, 0], dtype=np.uint32)
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("version" in e for e in r.errors)

    @pytest.mark.parametrize(
        "col,data,keyword",
        [
            ("afferent_section_pos", np.array([0.5, 1.5, -0.1], dtype=np.float64), "outside"),
            ("spine_length", np.array([1.0, 0.0, -0.5], dtype=np.float64), "<= 0"),
            ("afferent_segment_offset", np.array([0.1, -0.5, 0.3], dtype=np.float64), "negative"),
        ],
        ids=["section-pos-range", "spine-length-zero", "offset-negative"],
    )
    def test_range_validation(self, tmp_path, col, data, keyword):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp[col]
            grp.create_dataset(col, data=data)
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any(keyword in e for e in r.errors)

    @pytest.mark.parametrize(
        "col",
        [
            "spine_volume",
            "spine_neck_diameter",
        ],
    )
    def test_optional_positive_column_invalid(self, tmp_path, col):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            grp.create_dataset(col, data=np.array([0.5, -0.1, 0.0], dtype=np.float64))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any(col in e and "<= 0" in e for e in r.errors)

    @pytest.mark.parametrize(
        "col,bad_val,keyword",
        [
            ("afferent_surface_x", np.nan, "NaN"),
            ("afferent_surface_y", np.inf, "Inf"),
        ],
        ids=["nan-in-float", "inf-in-float"],
    )
    def test_bad_float_values(self, tmp_path, col, bad_val, keyword):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp[col]
            grp.create_dataset(col, data=np.array([1.0, bad_val, 3.0]))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any(keyword in e for e in r.errors)

    def test_spine_morphology_invalid_reference(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_morphology"]
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "spine_morphology",
                data=np.array(["nonexistent_group"] * 3, dtype=object),
                dtype=dt,
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("nonexistent_group" in e or "not in" in e for e in r.errors)

    def test_empty_edges_group(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_EDGES}/{NEURON_NAME}"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("empty" in e for e in r.errors)

    def test_inconsistent_dataset_lengths(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_length"]
            grp.create_dataset("spine_length", data=np.array([1.0, 2.0]))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("inconsistent lengths" in e for e in r.errors)

    def test_wrong_dtype_for_integer_column(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_section_id"]
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "afferent_section_id",
                data=np.array(["a", "b", "c"], dtype=object),
                dtype=dt,
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("integer" in e for e in r.errors)

    def test_edges_entry_is_dataset_not_group(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            h5[GRP_EDGES].create_dataset(NEURON_NAME, data=np.zeros(3))
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("not a group" in e for e in r.errors)

    def test_edges_metadata_missing_version_attr(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            meta = h5[f"{GRP_EDGES}/{NEURON_NAME}/{GRP_METADATA}"]
            del meta.attrs[ATT_VERSION]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("version" in e for e in r.errors)

    def test_spine_type_invalid_values(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "spine_type",
                data=np.array(["thin", "invalid_type", "mushroom"], dtype=object),
                dtype=dt,
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("invalid_type" in e for e in r.errors)


class TestSpinesGroup:
    """Tests for /spines group validation."""

    def test_missing_skeletons_subgroup(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("skeletons" in e for e in r.errors)

    def test_meshes_optional(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_SPINES}/{GRP_MESHES}"]
        r = validate_morph_with_spines_file(filepath)
        assert r.is_valid
        assert any("not present" in i for i in r.info)

    def test_skeleton_wrong_points_shape(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            skel = h5[f"{GRP_SPINES}/{GRP_SKELETONS}/{SPINE_MORPH_NAME}"]
            del skel["points"]
            skel.create_dataset("points", data=np.ones((5, 3), dtype=np.float32))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("shape (N, 4)" in e for e in r.errors)

    def test_mesh_missing_offsets(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        mesh = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
        with h5py.File(filepath, "a") as h5:
            del h5[f"{mesh}/{GRP_OFFSETS}"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("offsets" in e for e in r.errors)

    @pytest.mark.parametrize(
        "bad_val,keyword",
        [
            (np.nan, "NaN"),
            (np.inf, "Inf"),
        ],
        ids=["vertices-nan", "vertices-inf"],
    )
    def test_mesh_vertices_bad_values(self, tmp_path, bad_val, keyword):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"]
            del grp[GRP_VERTICES]
            data = np.ones((9, 3), dtype=np.float32)
            data[2, 1] = bad_val
            grp.create_dataset(GRP_VERTICES, data=data, compression="gzip")
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any(keyword in e for e in r.errors)

    @pytest.mark.parametrize(
        "dataset,bad_shape,keyword",
        [
            (GRP_VERTICES, (9, 2), "shape (N, 3)"),
            (GRP_TRIANGLES, (3, 4), "shape (M, 3)"),
            (GRP_OFFSETS, (4, 4), "shape"),
        ],
        ids=["vertices-shape", "triangles-shape", "offsets-shape"],
    )
    def test_mesh_wrong_shape(self, tmp_path, dataset, bad_shape, keyword):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        dtype = np.float32 if dataset == GRP_VERTICES else np.int32
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"]
            del grp[dataset]
            grp.create_dataset(
                dataset,
                data=np.ones(bad_shape, dtype=dtype),
                compression="gzip",
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any(keyword in e for e in r.errors)

    def test_triangle_index_out_of_bounds(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"]
            del grp[GRP_TRIANGLES]
            grp.create_dataset(
                GRP_TRIANGLES,
                data=np.array([[0, 1, 5], [0, 1, 2], [0, 1, 2]], dtype=np.int32),
                compression="gzip",
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("local vertex count" in e for e in r.errors)

    @pytest.mark.parametrize(
        "offsets_data",
        [
            np.array([[0, 0], [3, 1], [6, 2], [3, 3]], dtype=np.int32),
            np.array([[0, 0], [3, 2], [6, 1], [9, 3]], dtype=np.int32),
        ],
        ids=["vertex-offsets-backward", "triangle-offsets-backward"],
    )
    def test_offsets_non_decreasing(self, tmp_path, offsets_data):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"]
            del grp[GRP_OFFSETS]
            grp.create_dataset(GRP_OFFSETS, data=offsets_data, compression="gzip")
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("non-decreasing" in e for e in r.errors)

    def test_offsets_first_row_not_zero(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"]
            del grp[GRP_OFFSETS]
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.array([[1, 1], [4, 2], [7, 3], [10, 4]], dtype=np.int32),
                compression="gzip",
            )
            del grp[GRP_VERTICES]
            grp.create_dataset(
                GRP_VERTICES,
                data=np.zeros((10, 3), dtype=np.float32),
                compression="gzip",
            )
            del grp[GRP_TRIANGLES]
            grp.create_dataset(
                GRP_TRIANGLES,
                data=np.array([[0, 1, 2]] * 4, dtype=np.int32),
                compression="gzip",
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert r.is_valid
        assert any("first row" in w for w in r.warnings)

    @pytest.mark.parametrize(
        "subgroup",
        [
            GRP_SKELETONS,
            GRP_MESHES,
        ],
        ids=["skeleton-is-dataset", "mesh-is-dataset"],
    )
    def test_entry_is_dataset_not_group(self, tmp_path, subgroup):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            parent = h5[f"{GRP_SPINES}/{subgroup}"]
            del parent[SPINE_MORPH_NAME]
            parent.create_dataset(SPINE_MORPH_NAME, data=np.zeros(3))
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("not a group" in e for e in r.errors)

    def test_offsets_single_row(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"]
            del grp[GRP_OFFSETS]
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.array([[0, 0]], dtype=np.int32),
                compression="gzip",
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("at least 2 rows" in e for e in r.errors)


class TestSomaGroup:
    """Tests for /soma group validation."""

    def test_soma_optional(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5["soma"]
        r = validate_morph_with_spines_file(filepath)
        assert r.is_valid
        assert any("soma" in i and "not present" in i for i in r.info)

    def test_soma_missing_vertices(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            del h5[f"soma/{GRP_MESHES}/{NEURON_NAME}/{GRP_VERTICES}"]
        r = validate_morph_with_spines_file(filepath)
        assert not r.is_valid
        assert any("vertices" in e for e in r.errors)

    @pytest.mark.parametrize(
        "dataset,bad_shape,keyword",
        [
            (GRP_VERTICES, (4, 2), "shape (N, 3)"),
            (GRP_TRIANGLES, (2, 4), "shape (M, 3)"),
        ],
        ids=["soma-vertices-shape", "soma-triangles-shape"],
    )
    def test_soma_mesh_wrong_shape(self, tmp_path, dataset, bad_shape, keyword):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        dtype = np.float32 if dataset == GRP_VERTICES else np.int32
        with h5py.File(filepath, "a") as h5:
            soma_path = f"soma/{GRP_MESHES}/{NEURON_NAME}"
            del h5[f"{soma_path}/{dataset}"]
            h5[soma_path].create_dataset(dataset, data=np.ones(bad_shape, dtype=dtype))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("soma" in e and keyword in e for e in r.errors)


class TestCrossGroupIntegrity:
    """Tests for cross-group referential integrity."""

    def test_edges_neuron_not_in_morphology(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            h5.move(f"{GRP_MORPH}/{NEURON_NAME}", f"{GRP_MORPH}/other_neuron")
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("not in /morphology" in e for e in r.errors)

    def test_spine_id_exceeds_skeleton_root_sections(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_id"]
            grp.create_dataset("spine_id", data=np.array([0, 1, 99], dtype=np.uint32))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("spine_id=99" in e and "skeleton" in e for e in r.errors)

    def test_spine_id_exceeds_mesh_offsets(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            skel = h5[f"{GRP_SPINES}/{GRP_SKELETONS}/{SPINE_MORPH_NAME}"]
            del skel["structure"]
            skel.create_dataset(
                "structure",
                data=np.array(
                    [
                        [0, 2, -1],
                        [2, 2, -1],
                        [4, 2, -1],
                        [5, 1, -1],
                        [6, 1, -1],
                    ],
                    dtype=np.int32,
                ),
            )
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_id"]
            grp.create_dataset("spine_id", data=np.array([0, 1, 4], dtype=np.uint32))
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("spine_id=4" in e and "mesh" in e for e in r.errors)

    def test_spine_morphology_missing_mesh_warning(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            src = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            h5.move(src, f"{GRP_SPINES}/{GRP_MESHES}/other_name")
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert any("spine_morphology" in w and "meshes" in w for w in r.warnings)

    def test_afferent_section_id_exceeds_morphology(self, tmp_path):
        filepath = write_minimal_valid_file(tmp_path / "test.h5")
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_section_id"]
            grp.create_dataset(
                "afferent_section_id",
                data=np.array([0, 0, 5], dtype=np.uint32),
            )
        r = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not r.is_valid
        assert any("afferent_section_id" in e for e in r.errors)


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_initial_state(self):
        r = ValidationResult()
        assert r.is_valid is True
        assert r.data_integrity_checked is False
        assert r.errors == []
        assert r.warnings == []
        assert r.info == []

    def test_add_error_marks_invalid(self):
        r = ValidationResult()
        r.add_error("something went wrong")
        assert not r.is_valid
        assert "something went wrong" in r.errors

    def test_add_warning_keeps_valid(self):
        r = ValidationResult()
        r.add_warning("minor issue")
        assert r.is_valid
        assert "minor issue" in r.warnings

    def test_merge(self):
        r1 = ValidationResult()
        r1.add_info("info1")
        r2 = ValidationResult()
        r2.add_error("error1")
        r2.add_warning("warn1")
        r1.merge(r2)
        assert not r1.is_valid
        assert "error1" in r1.errors
        assert "warn1" in r1.warnings
        assert "info1" in r1.info

    def test_str_structure_only(self):
        r = ValidationResult()
        assert "structure only" in str(r)

    def test_str_with_data_integrity(self):
        r = ValidationResult(data_integrity_checked=True)
        assert "data integrity" in str(r)

    def test_str_all_sections(self):
        r = ValidationResult()
        r.add_info("loaded file")
        r.add_warning("heads up")
        r.add_error("broken")
        output = str(r)
        assert "Info (1)" in output
        assert "loaded file" in output
        assert "Warnings (1)" in output
        assert "heads up" in output
        assert "Errors (1)" in output
        assert "broken" in output
        assert "INVALID" in output
