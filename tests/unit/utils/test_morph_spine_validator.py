"""Unit tests for morph_spines.utils.morph_spine_validator."""

import h5py
import numpy as np

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
        """Nonexistent file produces 'not found' error."""
        path = tmp_path / "nonexistent.h5"
        result = validate_morph_with_spines_file(path)
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_not_hdf5(self, tmp_path):
        """Plain text file produces 'Cannot open' error."""
        path = tmp_path / "bad.h5"
        path.write_text("not an hdf5 file")
        result = validate_morph_with_spines_file(path)
        assert not result.is_valid
        assert any("Cannot open" in e for e in result.errors)

    def test_valid_file_structure_only(self, tmp_path):
        """Valid file passes structural validation."""
        filepath = tmp_path / "valid.h5"
        write_minimal_valid_file(filepath)
        result = validate_morph_with_spines_file(filepath)
        assert result.is_valid
        assert not result.data_integrity_checked

    def test_valid_file_with_data_integrity(self, tmp_path):
        """Valid file passes with data integrity checks enabled."""
        filepath = tmp_path / "valid.h5"
        write_minimal_valid_file(filepath)
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert result.is_valid
        assert result.data_integrity_checked

    def test_path_is_directory(self, tmp_path):
        """Validating a directory path reports error."""
        result = validate_morph_with_spines_file(tmp_path)
        assert not result.is_valid
        assert any("not a file" in e for e in result.errors)

    def test_unexpected_top_level_groups(self, tmp_path):
        """Unexpected top-level groups produce warning."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            h5.create_group("unexpected_group")
        result = validate_morph_with_spines_file(filepath)
        assert result.is_valid
        assert any("unexpected_group" in w for w in result.warnings)


class TestMorphologyGroup:
    """Tests for /morphology group validation."""

    def test_missing_morphology_group(self, tmp_path):
        """File without /morphology reports error."""
        filepath = tmp_path / "no_morph.h5"
        with h5py.File(filepath, "w") as h5:
            h5.create_group(GRP_EDGES)
            h5.create_group(GRP_SPINES)
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("/morphology" in e for e in result.errors)

    def test_missing_points_dataset(self, tmp_path):
        """Missing points dataset in morphology reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_MORPH}/{NEURON_NAME}/points"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("missing 'points'" in e for e in result.errors)

    def test_wrong_points_shape(self, tmp_path):
        """Points with wrong shape (3,3) reports shape error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            grp.create_dataset(
                "points",
                data=np.ones((3, 3), dtype=np.float32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("shape (N, 4)" in e for e in result.errors)

    def test_points_not_float32_warning(self, tmp_path):
        """Points with float64 dtype produces warning."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            grp.create_dataset(
                "points",
                data=np.ones((3, 4), dtype=np.float64),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert result.is_valid
        assert any("float32" in w for w in result.warnings)

    def test_points_with_nan(self, tmp_path):
        """Points containing NaN values reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            data = np.ones((3, 4), dtype=np.float32)
            data[1, 2] = np.nan
            grp.create_dataset("points", data=data)
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    def test_empty_morphology_group(self, tmp_path):
        """Empty /morphology group reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_MORPH}/{NEURON_NAME}"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("empty" in e for e in result.errors)

    def test_structure_wrong_shape(self, tmp_path):
        """Structure with shape (3,2) reports shape error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["structure"]
            grp.create_dataset(
                "structure",
                data=np.ones((3, 2), dtype=np.int32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("shape (M, 3)" in e for e in result.errors)

    def test_points_with_inf(self, tmp_path):
        """Points containing Inf values reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            data = np.ones((3, 4), dtype=np.float32)
            data[0, 0] = np.inf
            grp.create_dataset("points", data=data)
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("Inf" in e for e in result.errors)

    def test_empty_points(self, tmp_path):
        """Empty points dataset reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["points"]
            grp.create_dataset(
                "points",
                data=np.empty((0, 4), dtype=np.float32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("empty" in e for e in result.errors)

    def test_empty_structure(self, tmp_path):
        """Empty structure dataset reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            del grp["structure"]
            grp.create_dataset(
                "structure",
                data=np.empty((0, 3), dtype=np.int32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("empty" in e for e in result.errors)

    def test_morphology_entry_is_dataset_not_group(self, tmp_path):
        """A dataset instead of group under /morphology errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_MORPH}/{NEURON_NAME}"]
            h5[GRP_MORPH].create_dataset(NEURON_NAME, data=np.zeros(3))
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("not a group" in e for e in result.errors)


class TestEdgesGroup:
    """Tests for /edges group validation."""

    def test_missing_edges_group(self, tmp_path):
        """File without /edges reports error."""
        filepath = tmp_path / "no_edges.h5"
        with h5py.File(filepath, "w") as h5:
            h5.create_group(GRP_MORPH)
            h5.create_group(GRP_SPINES)
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("/edges" in e for e in result.errors)

    def test_missing_mandatory_column(self, tmp_path):
        """Missing spine_length column reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_EDGES}/{NEURON_NAME}/spine_length"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("spine_length" in e for e in result.errors)

    def test_unknown_column_warning(self, tmp_path):
        """Unknown column dataset produces warning."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            grp.create_dataset("unknown_col", data=np.zeros(3))
        result = validate_morph_with_spines_file(filepath)
        assert result.is_valid
        assert any("unknown_col" in w for w in result.warnings)

    def test_wrong_metadata_version(self, tmp_path):
        """Wrong spine table version reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            meta = h5[f"{GRP_EDGES}/{NEURON_NAME}/{GRP_METADATA}"]
            meta.attrs[ATT_VERSION] = np.array([99, 0], dtype=np.uint32)
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("version" in e for e in result.errors)

    def test_afferent_section_pos_out_of_range(self, tmp_path):
        """Section pos values outside [0,1] report error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_section_pos"]
            grp.create_dataset(
                "afferent_section_pos",
                data=np.array([0.5, 1.5, -0.1], dtype=np.float64),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("outside" in e for e in result.errors)

    def test_spine_length_zero(self, tmp_path):
        """Spine length with zero or negative values errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_length"]
            grp.create_dataset(
                "spine_length",
                data=np.array([1.0, 0.0, -0.5], dtype=np.float64),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("<= 0" in e for e in result.errors)

    def test_afferent_segment_offset_negative(self, tmp_path):
        """Negative segment offset values report error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_segment_offset"]
            grp.create_dataset(
                "afferent_segment_offset",
                data=np.array([0.1, -0.5, 0.3], dtype=np.float64),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("negative" in e for e in result.errors)

    def test_spine_morphology_invalid_reference(self, tmp_path):
        """spine_morphology referencing nonexistent group."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_morphology"]
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "spine_morphology",
                data=np.array(["nonexistent_group"] * 3, dtype=object),
                dtype=dt,
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("nonexistent_group" in e or "not in" in e for e in result.errors)

    def test_empty_edges_group(self, tmp_path):
        """Empty /edges group reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_EDGES}/{NEURON_NAME}"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("empty" in e for e in result.errors)

    def test_inconsistent_dataset_lengths(self, tmp_path):
        """Datasets with different lengths report error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_length"]
            grp.create_dataset(
                "spine_length",
                data=np.array([1.0, 2.0], dtype=np.float64),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("inconsistent lengths" in e for e in result.errors)

    def test_nan_in_float_column(self, tmp_path):
        """NaN in float column reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_surface_x"]
            data = np.array([1.0, np.nan, 3.0], dtype=np.float64)
            grp.create_dataset("afferent_surface_x", data=data)
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    def test_inf_in_float_column(self, tmp_path):
        """Inf in float column reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_surface_y"]
            data = np.array([1.0, np.inf, 3.0], dtype=np.float64)
            grp.create_dataset("afferent_surface_y", data=data)
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("Inf" in e for e in result.errors)

    def test_wrong_dtype_for_integer_column(self, tmp_path):
        """String dtype for integer column reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["afferent_section_id"]
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "afferent_section_id",
                data=np.array(["a", "b", "c"], dtype=object),
                dtype=dt,
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("integer" in e for e in result.errors)

    def test_edges_entry_is_dataset_not_group(self, tmp_path):
        """A dataset instead of group under /edges errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            h5[GRP_EDGES].create_dataset(NEURON_NAME, data=np.zeros(3))
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("not a group" in e for e in result.errors)

    def test_edges_metadata_missing_version_attr(self, tmp_path):
        """Metadata group without version attribute errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            meta = h5[f"{GRP_EDGES}/{NEURON_NAME}/{GRP_METADATA}"]
            del meta.attrs[ATT_VERSION]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("version" in e for e in result.errors)


class TestSpinesGroup:
    """Tests for /spines group validation."""

    def test_missing_spines_group(self, tmp_path):
        """File without /spines reports error."""
        filepath = tmp_path / "no_spines.h5"
        with h5py.File(filepath, "w") as h5:
            h5.create_group(GRP_MORPH)
            h5.create_group(GRP_EDGES)
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("/spines" in e for e in result.errors)

    def test_missing_skeletons_subgroup(self, tmp_path):
        """Missing /spines/skeletons reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("skeletons" in e for e in result.errors)

    def test_meshes_optional(self, tmp_path):
        """Missing /spines/meshes is valid with info message."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"{GRP_SPINES}/{GRP_MESHES}"]
        result = validate_morph_with_spines_file(filepath)
        assert result.is_valid
        assert any("not present" in i for i in result.info)

    def test_skeleton_wrong_points_shape(self, tmp_path):
        """Skeleton points with shape (5,3) reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            skel_path = f"{GRP_SPINES}/{GRP_SKELETONS}/{SPINE_MORPH_NAME}"
            grp = h5[skel_path]
            del grp["points"]
            grp.create_dataset(
                "points",
                data=np.ones((5, 3), dtype=np.float32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("shape (N, 4)" in e for e in result.errors)

    def test_mesh_missing_offsets(self, tmp_path):
        """Missing offsets dataset in mesh reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            del h5[f"{mesh_path}/{GRP_OFFSETS}"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("offsets" in e for e in result.errors)

    def test_mesh_vertices_nan(self, tmp_path):
        """NaN in mesh vertices reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_VERTICES]
            data = np.ones((9, 3), dtype=np.float32)
            data[2, 1] = np.nan
            grp.create_dataset(GRP_VERTICES, data=data, compression="gzip")
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)

    def test_mesh_vertices_wrong_shape(self, tmp_path):
        """Mesh vertices with shape (9,2) reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_VERTICES]
            grp.create_dataset(
                GRP_VERTICES,
                data=np.ones((9, 2), dtype=np.float32),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("shape (N, 3)" in e for e in result.errors)

    def test_mesh_triangles_wrong_shape(self, tmp_path):
        """Mesh triangles with shape (3,4) reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_TRIANGLES]
            grp.create_dataset(
                GRP_TRIANGLES,
                data=np.ones((3, 4), dtype=np.int32),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("shape (M, 3)" in e for e in result.errors)

    def test_triangle_index_out_of_bounds(self, tmp_path):
        """Triangle index exceeding local vertex count."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_TRIANGLES]
            # Each spine has 3 local vertices; index 5 is OOB
            grp.create_dataset(
                GRP_TRIANGLES,
                data=np.array(
                    [[0, 1, 5], [0, 1, 2], [0, 1, 2]],
                    dtype=np.int32,
                ),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("local vertex count" in e for e in result.errors)

    def test_offsets_wrong_shape(self, tmp_path):
        """Offsets with shape (4,4) reports shape error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_OFFSETS]
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.ones((4, 4), dtype=np.int32),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("shape" in e for e in result.errors)

    def test_offsets_non_decreasing(self, tmp_path):
        """Offsets going backwards reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_OFFSETS]
            # Non-decreasing violation: 6 -> 3
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.array(
                    [[0, 0], [3, 1], [6, 2], [3, 3]],
                    dtype=np.int32,
                ),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("non-decreasing" in e for e in result.errors)

    def test_offsets_first_row_not_zero(self, tmp_path):
        """Offsets with non-zero first row produces warning."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_OFFSETS]
            # First row not zero
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.array(
                    [[1, 1], [4, 2], [7, 3], [10, 4]],
                    dtype=np.int32,
                ),
                compression="gzip",
            )
            # Also need extra vertices/triangles to match
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
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert result.is_valid
        assert any("first row" in w for w in result.warnings)

    def test_skeleton_entry_is_dataset_not_group(self, tmp_path):
        """A dataset instead of group under skeletons errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            skel_parent = h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
            del skel_parent[SPINE_MORPH_NAME]
            skel_parent.create_dataset(SPINE_MORPH_NAME, data=np.zeros(3))
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("not a group" in e for e in result.errors)

    def test_mesh_entry_is_dataset_not_group(self, tmp_path):
        """A dataset instead of group under meshes errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_parent = h5[f"{GRP_SPINES}/{GRP_MESHES}"]
            del mesh_parent[SPINE_MORPH_NAME]
            mesh_parent.create_dataset(SPINE_MORPH_NAME, data=np.zeros(3))
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("not a group" in e for e in result.errors)

    def test_vertices_with_inf(self, tmp_path):
        """Mesh vertices with Inf reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_VERTICES]
            data = np.ones((9, 3), dtype=np.float32)
            data[0, 0] = np.inf
            grp.create_dataset(GRP_VERTICES, data=data, compression="gzip")
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("Inf" in e for e in result.errors)

    def test_offsets_single_row(self, tmp_path):
        """Offsets with only 1 row reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_OFFSETS]
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.array([[0, 0]], dtype=np.int32),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("at least 2 rows" in e for e in result.errors)

    def test_triangle_offsets_non_decreasing(self, tmp_path):
        """Triangle offsets going backwards reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            grp = h5[mesh_path]
            del grp[GRP_OFFSETS]
            # Triangle offsets (col 1) go backwards
            grp.create_dataset(
                GRP_OFFSETS,
                data=np.array(
                    [[0, 0], [3, 2], [6, 1], [9, 3]],
                    dtype=np.int32,
                ),
                compression="gzip",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("non-decreasing" in e for e in result.errors)


class TestSomaGroup:
    """Tests for /soma group validation."""

    def test_soma_optional(self, tmp_path):
        """Missing /soma is valid with info message."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5["soma"]
        result = validate_morph_with_spines_file(filepath)
        assert result.is_valid
        assert any("soma" in i and "not present" in i for i in result.info)

    def test_soma_missing_vertices(self, tmp_path):
        """Missing vertices in soma mesh reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            del h5[f"soma/{GRP_MESHES}/{NEURON_NAME}/{GRP_VERTICES}"]
        result = validate_morph_with_spines_file(filepath)
        assert not result.is_valid
        assert any("vertices" in e for e in result.errors)

    def test_soma_vertices_wrong_shape(self, tmp_path):
        """Soma vertices with shape (4,2) reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            soma_path = f"soma/{GRP_MESHES}/{NEURON_NAME}"
            del h5[f"{soma_path}/{GRP_VERTICES}"]
            h5[soma_path].create_dataset(
                GRP_VERTICES,
                data=np.ones((4, 2), dtype=np.float32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("soma" in e and "vertices" in e and "shape (N, 3)" in e for e in result.errors)

    def test_soma_triangles_wrong_shape(self, tmp_path):
        """Soma triangles with shape (2,4) reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            soma_path = f"soma/{GRP_MESHES}/{NEURON_NAME}"
            del h5[f"{soma_path}/{GRP_TRIANGLES}"]
            h5[soma_path].create_dataset(
                GRP_TRIANGLES,
                data=np.ones((2, 4), dtype=np.int32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("soma" in e and "triangles" in e and "shape (M, 3)" in e for e in result.errors)


class TestCrossGroupIntegrity:
    """Tests for cross-group referential integrity."""

    def test_edges_neuron_not_in_morphology(self, tmp_path):
        """Edges neuron not in /morphology reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            h5.move(
                f"{GRP_MORPH}/{NEURON_NAME}",
                f"{GRP_MORPH}/other_neuron",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("not in /morphology" in e for e in result.errors)

    def test_spine_id_exceeds_skeleton_root_sections(self, tmp_path):
        """spine_id exceeding root sections reports error."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_id"]
            grp.create_dataset(
                "spine_id",
                data=np.array([0, 1, 99], dtype=np.uint32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("spine_id=99" in e and "skeleton" in e for e in result.errors)

    def test_spine_id_exceeds_mesh_offsets(self, tmp_path):
        """spine_id exceeding mesh offset count errors."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            # Add more root sections to skeleton (5 roots)
            skel_path = f"{GRP_SPINES}/{GRP_SKELETONS}/{SPINE_MORPH_NAME}"
            skel_grp = h5[skel_path]
            del skel_grp["structure"]
            skel_grp.create_dataset(
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
            # Set spine_id to [0, 1, 4] — 4 exceeds mesh
            grp = h5[f"{GRP_EDGES}/{NEURON_NAME}"]
            del grp["spine_id"]
            grp.create_dataset(
                "spine_id",
                data=np.array([0, 1, 4], dtype=np.uint32),
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert not result.is_valid
        assert any("spine_id=4" in e and "mesh" in e for e in result.errors)

    def test_spine_morphology_missing_mesh_warning(self, tmp_path):
        """spine_morphology referencing missing mesh warns."""
        filepath = tmp_path / "test.h5"
        write_minimal_valid_file(filepath)
        with h5py.File(filepath, "a") as h5:
            mesh_path = f"{GRP_SPINES}/{GRP_MESHES}/{SPINE_MORPH_NAME}"
            h5.move(
                mesh_path,
                f"{GRP_SPINES}/{GRP_MESHES}/other_name",
            )
        result = validate_morph_with_spines_file(filepath, check_data_integrity=True)
        assert any("spine_morphology" in w and "meshes" in w for w in result.warnings)


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_initial_state(self):
        """New ValidationResult has correct defaults."""
        r = ValidationResult()
        assert r.is_valid is True
        assert r.data_integrity_checked is False
        assert r.errors == []
        assert r.warnings == []
        assert r.info == []

    def test_add_error_marks_invalid(self):
        """add_error sets is_valid to False."""
        r = ValidationResult()
        r.add_error("something went wrong")
        assert not r.is_valid
        assert "something went wrong" in r.errors

    def test_add_warning_keeps_valid(self):
        """add_warning does not change is_valid."""
        r = ValidationResult()
        r.add_warning("minor issue")
        assert r.is_valid
        assert "minor issue" in r.warnings

    def test_merge(self):
        """merge combines errors, warnings, and info."""
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
        """__str__ shows 'structure only' when no data check."""
        r = ValidationResult()
        assert "structure only" in str(r)

    def test_str_with_data_integrity(self):
        """__str__ shows 'data integrity' when checked."""
        r = ValidationResult(data_integrity_checked=True)
        assert "data integrity" in str(r)

    def test_str_all_sections(self):
        """__str__ includes info, warnings, and errors."""
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
