import h5py
import numpy as np
import pandas as pd
import pytest

from morph_spines.core.h5_schema import (
    ATT_VERSION,
    GRP_EDGES,
    GRP_MESHES,
    GRP_METADATA,
    GRP_MORPH,
    GRP_OFFSETS,
    GRP_SKELETONS,
    GRP_SOMA,
    GRP_SPINES,
    GRP_TRIANGLES,
    GRP_VERTICES,
    MANDATORY_COLUMNS,
    SPINE_TABLE_VER_H5_DATASETS,
)
from morph_spines.utils.morph_spine_writer import (
    validate_spine_table,
    write_morphology,
    write_soma_mesh,
    write_spine_meshes,
    write_spine_skeletons,
    write_spine_table,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def valid_spine_table():
    """Create a minimal valid spine table with all mandatory columns."""
    n = 3
    return pd.DataFrame(
        {
            "afferent_surface_x": np.random.rand(n),
            "afferent_surface_y": np.random.rand(n),
            "afferent_surface_z": np.random.rand(n),
            "afferent_center_x": np.random.rand(n),
            "afferent_center_y": np.random.rand(n),
            "afferent_center_z": np.random.rand(n),
            "spine_morphology": ["morph_a"] * n,
            "spine_id": np.array([0, 1, 2], dtype=np.uint32),
            "spine_length": np.random.rand(n),
            "spine_orientation_vector_x": np.random.rand(n),
            "spine_orientation_vector_y": np.random.rand(n),
            "spine_orientation_vector_z": np.random.rand(n),
            "spine_rotation_x": np.random.rand(n),
            "spine_rotation_y": np.random.rand(n),
            "spine_rotation_z": np.random.rand(n),
            "spine_rotation_w": np.random.rand(n),
            "afferent_section_id": np.array([1, 2, 3], dtype=np.uint32),
            "afferent_segment_id": np.array([0, 1, 2], dtype=np.int32),
            "afferent_segment_offset": np.random.rand(n),
            "afferent_section_pos": np.random.rand(n),
        }
    )


@pytest.fixture
def sample_vertices():
    return np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)


@pytest.fixture
def sample_triangles():
    return np.array([[0, 1, 2]], dtype=np.int32)


@pytest.fixture
def sample_points():
    return np.array(
        [
            [0.0, 0.0, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.4],
            [2.0, 0.0, 0.0, 0.3],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def sample_structure():
    return np.array([[0, 2, -1]], dtype=np.int32)


# =============================================================================
# validate_spine_table
# =============================================================================


class TestValidateSpineTable:
    def test_valid_table_passes(self, valid_spine_table):
        # Should not raise
        validate_spine_table(valid_spine_table)

    def test_valid_table_with_optional_columns(self, valid_spine_table):
        valid_spine_table["spine_volume"] = np.random.rand(len(valid_spine_table))
        valid_spine_table["spine_type"] = ["thin"] * len(valid_spine_table)
        validate_spine_table(valid_spine_table)

    def test_missing_mandatory_columns(self, valid_spine_table):
        df = valid_spine_table.drop(columns=["spine_id", "spine_length"])
        with pytest.raises(ValueError, match="Missing mandatory columns"):
            validate_spine_table(df)

    def test_unknown_columns(self, valid_spine_table):
        valid_spine_table["unknown_col"] = 1.0
        with pytest.raises(ValueError, match="Unknown columns"):
            validate_spine_table(valid_spine_table)

    def test_wrong_dtype_string_column(self, valid_spine_table):
        valid_spine_table["spine_morphology"] = np.array([1, 2, 3], dtype=np.int32)
        with pytest.raises(ValueError, match="expected string type"):
            validate_spine_table(valid_spine_table)

    def test_wrong_dtype_float_column(self, valid_spine_table):
        valid_spine_table["spine_length"] = ["not", "a", "number"]
        with pytest.raises(ValueError, match="expected numeric"):
            validate_spine_table(valid_spine_table)

    def test_wrong_dtype_unsigned_int_column(self, valid_spine_table):
        valid_spine_table["spine_id"] = ["not", "an", "int"]
        with pytest.raises(ValueError, match="expected integer type"):
            validate_spine_table(valid_spine_table)

    def test_wrong_dtype_signed_int_column(self, valid_spine_table):
        valid_spine_table["afferent_segment_id"] = ["a", "b", "c"]
        with pytest.raises(ValueError, match="expected signed integer type"):
            validate_spine_table(valid_spine_table)

    def test_float_accepted_for_int_columns(self, valid_spine_table):
        # pandas often stores int columns as float64 (e.g., with NaN)
        valid_spine_table["spine_id"] = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        valid_spine_table["afferent_section_id"] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        valid_spine_table["afferent_segment_id"] = np.array([0.0, 1.0, 2.0], dtype=np.float64)
        # Should not raise
        validate_spine_table(valid_spine_table)

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="Missing mandatory columns"):
            validate_spine_table(df)

    def test_multiple_errors_reported(self):
        df = pd.DataFrame({"unknown_col": [1, 2], "spine_morphology": [1, 2]})
        with pytest.raises(ValueError, match="Spine table validation failed"):
            validate_spine_table(df)

    def test_ndim_check_with_multidimensional_column(self, valid_spine_table):
        # Force a 2D array into a column via object dtype wrapping
        valid_spine_table["afferent_surface_x"] = pd.array(
            [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])],
            dtype=object,
        )
        with pytest.raises(ValueError, match="Spine table validation failed"):
            validate_spine_table(valid_spine_table)


# =============================================================================
# write_spine_table
# =============================================================================


class TestWriteSpineTable:
    def test_write_creates_file(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        assert f.exists()
        with h5py.File(f, "r") as h5:
            assert f"{GRP_EDGES}/neuron_01" in h5
            assert f"{GRP_EDGES}/neuron_01/{GRP_METADATA}" in h5

    def test_write_version_metadata(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        with h5py.File(f, "r") as h5:
            version = h5[f"{GRP_EDGES}/neuron_01/{GRP_METADATA}"].attrs[ATT_VERSION]
            assert tuple(version) == SPINE_TABLE_VER_H5_DATASETS

    def test_write_columns_as_datasets(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_EDGES}/neuron_01"]
            for col in MANDATORY_COLUMNS:
                assert col in grp, f"Column '{col}' not found in H5 group"

    def test_write_float_data_roundtrip(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_EDGES}/neuron_01"]
            np.testing.assert_array_almost_equal(
                grp["afferent_surface_x"][:],
                valid_spine_table["afferent_surface_x"].to_numpy(),
            )

    def test_write_string_data_roundtrip(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_EDGES}/neuron_01"]
            stored = grp["spine_morphology"][:].astype(str)
            expected = valid_spine_table["spine_morphology"].to_numpy().astype(str)
            np.testing.assert_array_equal(stored, expected)

    def test_write_appends_to_existing_file(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        # Create file with some existing data
        with h5py.File(f, "w") as h5:
            h5.create_group("other_group")

        write_spine_table(str(f), "neuron_01", valid_spine_table)

        with h5py.File(f, "r") as h5:
            assert "other_group" in h5
            assert f"{GRP_EDGES}/neuron_01" in h5

    def test_write_raises_if_group_exists(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        with pytest.raises(ValueError, match="already exists"):
            write_spine_table(str(f), "neuron_01", valid_spine_table)

    def test_write_multiple_neurons(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)
        write_spine_table(str(f), "neuron_02", valid_spine_table)

        with h5py.File(f, "r") as h5:
            assert f"{GRP_EDGES}/neuron_01" in h5
            assert f"{GRP_EDGES}/neuron_02" in h5

    def test_write_validates_before_writing(self, tmp_path):
        f = tmp_path / "output.h5"
        bad_df = pd.DataFrame({"unknown": [1, 2]})

        with pytest.raises(ValueError, match="Spine table validation failed"):
            write_spine_table(str(f), "neuron_01", bad_df)

        # File should not have been created
        assert not f.exists()


# =============================================================================
# write_morphology
# =============================================================================


class TestWriteMorphology:
    def test_write_creates_group(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_morphology(str(f), "neuron_01", sample_points, sample_structure)

        with h5py.File(f, "r") as h5:
            assert f"{GRP_MORPH}/neuron_01" in h5
            assert f"{GRP_MORPH}/neuron_01/points" in h5
            assert f"{GRP_MORPH}/neuron_01/structure" in h5
            assert f"{GRP_MORPH}/neuron_01/{GRP_METADATA}" in h5

    def test_write_metadata(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_morphology(str(f), "neuron_01", sample_points, sample_structure, cell_family=1)

        with h5py.File(f, "r") as h5:
            metadata = h5[f"{GRP_MORPH}/neuron_01/{GRP_METADATA}"]
            assert metadata.attrs["cell_family"] == 1
            assert tuple(metadata.attrs["version"]) == (1, 3)

    def test_write_data_roundtrip(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_morphology(str(f), "neuron_01", sample_points, sample_structure)

        with h5py.File(f, "r") as h5:
            np.testing.assert_array_almost_equal(
                h5[f"{GRP_MORPH}/neuron_01/points"][:], sample_points
            )
            np.testing.assert_array_equal(
                h5[f"{GRP_MORPH}/neuron_01/structure"][:], sample_structure
            )

    def test_write_raises_if_group_exists(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_morphology(str(f), "neuron_01", sample_points, sample_structure)

        with pytest.raises(ValueError, match="already exists"):
            write_morphology(str(f), "neuron_01", sample_points, sample_structure)


# =============================================================================
# write_soma_mesh
# =============================================================================


class TestWriteSomaMesh:
    def test_write_creates_group(self, tmp_path, sample_vertices, sample_triangles):
        f = tmp_path / "output.h5"
        write_soma_mesh(str(f), "neuron_01", sample_vertices, sample_triangles)

        with h5py.File(f, "r") as h5:
            assert f"{GRP_SOMA}/{GRP_MESHES}/neuron_01" in h5
            assert f"{GRP_SOMA}/{GRP_MESHES}/neuron_01/{GRP_VERTICES}" in h5
            assert f"{GRP_SOMA}/{GRP_MESHES}/neuron_01/{GRP_TRIANGLES}" in h5

    def test_write_data_roundtrip(self, tmp_path, sample_vertices, sample_triangles):
        f = tmp_path / "output.h5"
        write_soma_mesh(str(f), "neuron_01", sample_vertices, sample_triangles)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SOMA}/{GRP_MESHES}/neuron_01"]
            np.testing.assert_array_almost_equal(grp[GRP_VERTICES][:], sample_vertices)
            np.testing.assert_array_equal(grp[GRP_TRIANGLES][:], sample_triangles)

    def test_write_not_compressed(self, tmp_path, sample_vertices, sample_triangles):
        f = tmp_path / "output.h5"
        write_soma_mesh(str(f), "neuron_01", sample_vertices, sample_triangles)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SOMA}/{GRP_MESHES}/neuron_01"]
            assert grp[GRP_VERTICES].compression is None
            assert grp[GRP_TRIANGLES].compression is None

    def test_write_raises_if_group_exists(self, tmp_path, sample_vertices, sample_triangles):
        f = tmp_path / "output.h5"
        write_soma_mesh(str(f), "neuron_01", sample_vertices, sample_triangles)

        with pytest.raises(ValueError, match="already exists"):
            write_soma_mesh(str(f), "neuron_01", sample_vertices, sample_triangles)


# =============================================================================
# write_spine_meshes
# =============================================================================


class TestWriteSpineMeshes:
    @pytest.fixture
    def sample_offsets(self):
        return np.array([[0, 0], [3, 1]], dtype=np.int32)

    def test_write_creates_group(self, tmp_path, sample_vertices, sample_triangles, sample_offsets):
        f = tmp_path / "output.h5"
        write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, sample_offsets)

        with h5py.File(f, "r") as h5:
            assert f"{GRP_SPINES}/{GRP_MESHES}/morph_a" in h5
            assert f"{GRP_SPINES}/{GRP_MESHES}/morph_a/{GRP_VERTICES}" in h5
            assert f"{GRP_SPINES}/{GRP_MESHES}/morph_a/{GRP_TRIANGLES}" in h5
            assert f"{GRP_SPINES}/{GRP_MESHES}/morph_a/{GRP_OFFSETS}" in h5

    def test_write_data_roundtrip(
        self, tmp_path, sample_vertices, sample_triangles, sample_offsets
    ):
        f = tmp_path / "output.h5"
        write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, sample_offsets)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/morph_a"]
            np.testing.assert_array_almost_equal(grp[GRP_VERTICES][:], sample_vertices)
            np.testing.assert_array_equal(grp[GRP_TRIANGLES][:], sample_triangles)
            np.testing.assert_array_equal(grp[GRP_OFFSETS][:], sample_offsets)

    def test_write_compressed(self, tmp_path, sample_vertices, sample_triangles, sample_offsets):
        f = tmp_path / "output.h5"
        write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, sample_offsets)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/morph_a"]
            assert grp[GRP_VERTICES].compression == "gzip"
            assert grp[GRP_TRIANGLES].compression == "gzip"
            assert grp[GRP_OFFSETS].compression == "gzip"

    def test_write_with_head_neck_values(self, tmp_path, sample_vertices, sample_triangles):
        offsets = np.array([[0, 0, 0], [3, 1, 2]], dtype=np.int32)
        hn_values = np.array([0, 1], dtype=np.int32)
        f = tmp_path / "output.h5"
        write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, offsets, hn_values)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/morph_a"]
            assert "head_neck_values" in grp
            np.testing.assert_array_equal(grp["head_neck_values"][:], hn_values)

    def test_write_without_head_neck_values(
        self, tmp_path, sample_vertices, sample_triangles, sample_offsets
    ):
        f = tmp_path / "output.h5"
        write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, sample_offsets)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_MESHES}/morph_a"]
            assert "head_neck_values" not in grp

    def test_write_raises_if_group_exists(
        self, tmp_path, sample_vertices, sample_triangles, sample_offsets
    ):
        f = tmp_path / "output.h5"
        write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, sample_offsets)

        with pytest.raises(ValueError, match="already exists"):
            write_spine_meshes(str(f), "morph_a", sample_vertices, sample_triangles, sample_offsets)


# =============================================================================
# write_spine_skeletons
# =============================================================================


class TestWriteSpineSkeletons:
    def test_write_creates_group(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_spine_skeletons(str(f), "morph_a", sample_points, sample_structure)

        with h5py.File(f, "r") as h5:
            assert f"{GRP_SPINES}/{GRP_SKELETONS}/morph_a" in h5
            assert f"{GRP_SPINES}/{GRP_SKELETONS}/morph_a/points" in h5
            assert f"{GRP_SPINES}/{GRP_SKELETONS}/morph_a/structure" in h5
            assert f"{GRP_SPINES}/{GRP_SKELETONS}/morph_a/{GRP_METADATA}" in h5

    def test_write_metadata(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_spine_skeletons(str(f), "morph_a", sample_points, sample_structure)

        with h5py.File(f, "r") as h5:
            metadata = h5[f"{GRP_SPINES}/{GRP_SKELETONS}/morph_a/{GRP_METADATA}"]
            assert metadata.attrs["cell_family"] == 0
            assert tuple(metadata.attrs["version"]) == (1, 3)

    def test_write_data_roundtrip(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_spine_skeletons(str(f), "morph_a", sample_points, sample_structure)

        with h5py.File(f, "r") as h5:
            grp = h5[f"{GRP_SPINES}/{GRP_SKELETONS}/morph_a"]
            np.testing.assert_array_almost_equal(grp["points"][:], sample_points)
            np.testing.assert_array_equal(grp["structure"][:], sample_structure)

    def test_write_raises_if_group_exists(self, tmp_path, sample_points, sample_structure):
        f = tmp_path / "output.h5"
        write_spine_skeletons(str(f), "morph_a", sample_points, sample_structure)

        with pytest.raises(ValueError, match="already exists"):
            write_spine_skeletons(str(f), "morph_a", sample_points, sample_structure)


# =============================================================================
# Integration: write full file and read back with loader
# =============================================================================


class TestWriteReadRoundtrip:
    def test_spine_table_readable_by_loader(self, tmp_path, valid_spine_table):
        from morph_spines.utils.morph_spine_loader import load_spine_table

        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        loaded = load_spine_table(str(f), f"{GRP_EDGES}/neuron_01")

        assert set(loaded.columns) == set(valid_spine_table.columns)
        assert len(loaded) == len(valid_spine_table)
        for col in valid_spine_table.columns:
            loaded_arr = loaded[col].to_numpy()
            expected_arr = valid_spine_table[col].to_numpy()
            # String columns: compare as strings
            if expected_arr.dtype == object or expected_arr.dtype.kind in ("U", "S", "O"):
                np.testing.assert_array_equal(
                    loaded_arr.astype(str),
                    expected_arr.astype(str),
                )
            elif loaded_arr.dtype.kind in ("U", "S", "O"):
                np.testing.assert_array_equal(
                    loaded_arr.astype(str),
                    expected_arr.astype(str),
                )
            else:
                np.testing.assert_array_almost_equal(
                    loaded_arr.astype(float),
                    expected_arr.astype(float),
                )
