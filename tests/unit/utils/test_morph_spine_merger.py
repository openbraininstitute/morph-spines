"""Unit tests for morph_spines.utils.morph_spine_merger."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from morph_spines.core.h5_schema import (
    COL_SPINE_MORPH,
    GRP_EDGES,
    GRP_MESHES,
    GRP_MORPH,
    GRP_SKELETONS,
    GRP_SOMA,
    GRP_SPINES,
)
from morph_spines.utils.morph_spine_merger import merge_morphologies_with_spines
from morph_spines.utils.morph_spine_writer import (
    write_morphology,
    write_soma_mesh,
    write_spine_meshes,
    write_spine_skeletons,
    write_spine_table,
)

SAMPLE_POINTS = np.array(
    [[0.0, 0.0, 0.0, 0.5], [1.0, 0.0, 0.0, 0.4], [2.0, 0.0, 0.0, 0.3]], dtype=np.float64
)
SAMPLE_STRUCTURE = np.array([[0, 2, -1]], dtype=np.int32)
SAMPLE_VERTICES = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
SAMPLE_TRIANGLES = np.array([[0, 1, 2]], dtype=np.int32)
SAMPLE_OFFSETS = np.array([[0, 0], [3, 1]], dtype=np.int32)


def _make_spine_table(spine_morph_name: str, n_spines: int = 3) -> pd.DataFrame:
    """Create a minimal valid spine table with spine_morphology referencing the given name."""
    return pd.DataFrame(
        {
            "afferent_surface_x": np.zeros(n_spines),
            "afferent_surface_y": np.zeros(n_spines),
            "afferent_surface_z": np.zeros(n_spines),
            "afferent_center_x": np.zeros(n_spines),
            "afferent_center_y": np.zeros(n_spines),
            "afferent_center_z": np.zeros(n_spines),
            "spine_morphology": [spine_morph_name] * n_spines,
            "spine_id": np.arange(n_spines, dtype=np.uint32),
            "spine_length": np.ones(n_spines),
            "spine_orientation_vector_x": np.zeros(n_spines),
            "spine_orientation_vector_y": np.zeros(n_spines),
            "spine_orientation_vector_z": np.ones(n_spines),
            "spine_rotation_x": np.zeros(n_spines),
            "spine_rotation_y": np.zeros(n_spines),
            "spine_rotation_z": np.zeros(n_spines),
            "spine_rotation_w": np.ones(n_spines),
            "afferent_section_id": np.arange(1, n_spines + 1, dtype=np.uint32),
            "afferent_segment_id": np.arange(n_spines, dtype=np.int32),
            "afferent_segment_offset": np.zeros(n_spines),
            "afferent_section_pos": np.zeros(n_spines),
        }
    )


def _create_source_file(
    filepath: Path,
    neuron_name: str,
    *,
    spine_group_name: str | None = None,
    include_meshes: bool = True,
    n_spines: int = 3,
) -> None:
    """Create a complete morph-with-spines source file for testing."""
    spine_grp = spine_group_name if spine_group_name is not None else neuron_name

    write_morphology(str(filepath), neuron_name, SAMPLE_POINTS, SAMPLE_STRUCTURE)
    write_spine_table(str(filepath), neuron_name, _make_spine_table(spine_grp, n_spines))
    write_spine_skeletons(str(filepath), spine_grp, SAMPLE_POINTS, SAMPLE_STRUCTURE)

    if include_meshes:
        write_soma_mesh(str(filepath), neuron_name, SAMPLE_VERTICES, SAMPLE_TRIANGLES)
        write_spine_meshes(
            str(filepath), spine_grp, SAMPLE_VERTICES, SAMPLE_TRIANGLES, SAMPLE_OFFSETS
        )


class TestValidation:
    def test_empty_source_files(self, tmp_path):
        with pytest.raises(ValueError, match="source_files must not be empty"):
            merge_morphologies_with_spines([], tmp_path / "out.h5")

    def test_output_already_exists(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A")
        output = tmp_path / "out.h5"
        output.touch()

        with pytest.raises(FileExistsError):
            merge_morphologies_with_spines([src], output)

    def test_missing_morphology_group(self, tmp_path):
        src = tmp_path / "bad.h5"
        with h5py.File(src, "w") as h5:
            h5.create_group("other")

        with pytest.raises(ValueError, match="Invalid file: No /morphology group"):
            merge_morphologies_with_spines([src], tmp_path / "out.h5")

    def test_missing_spines_table(self, tmp_path):
        src = tmp_path / "src.h5"
        write_morphology(str(src), "neuron_A", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        # Add /spines/skeletons so validation gets past that check
        write_spine_skeletons(str(src), "neuron_A", SAMPLE_POINTS, SAMPLE_STRUCTURE)

        with pytest.raises(ValueError, match="Invalid file: No /edges/neuron_A/spine_morphology"):
            merge_morphologies_with_spines([src], tmp_path / "out.h5")

    def test_duplicate_destination_names(self, tmp_path):
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A")
        _create_source_file(src2, "neuron_B")

        rename = {(src1, "neuron_A"): "same", (src2, "neuron_B"): "same"}
        with pytest.raises(ValueError, match="Duplicate morphology destination name"):
            merge_morphologies_with_spines([src1, src2], tmp_path / "out.h5", rename_map=rename)

    def test_duplicate_names_without_rename(self, tmp_path):
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A")
        _create_source_file(src2, "neuron_A")

        with pytest.raises(ValueError, match="Duplicate morphology destination name"):
            merge_morphologies_with_spines([src1, src2], tmp_path / "out.h5")


class TestMergeNoRename:
    def test_single_file(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output)

        with h5py.File(output, "r") as h5:
            assert "neuron_A" in h5[GRP_MORPH]
            assert "neuron_A" in h5[GRP_EDGES]
            assert "neuron_A" in h5[f"{GRP_SOMA}/{GRP_MESHES}"]
            assert "neuron_A" in h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
            assert "neuron_A" in h5[f"{GRP_SPINES}/{GRP_MESHES}"]

    def test_multiple_files(self, tmp_path):
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A", n_spines=5)
        _create_source_file(src2, "neuron_B", n_spines=7)
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src1, src2], output)

        with h5py.File(output, "r") as h5:
            assert set(h5[GRP_MORPH].keys()) == {"neuron_A", "neuron_B"}
            # Data integrity: total spines preserved
            total = sum(len(h5[f"{GRP_EDGES}/{k}/spine_id"][:]) for k in h5[GRP_EDGES].keys())
            assert total == 12

    def test_preserves_data_byte_for_byte(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output)

        with h5py.File(output, "r") as h5:
            # Morphology skeleton
            np.testing.assert_array_equal(h5[f"{GRP_MORPH}/neuron_A/points"][:], SAMPLE_POINTS)
            np.testing.assert_array_equal(
                h5[f"{GRP_MORPH}/neuron_A/structure"][:], SAMPLE_STRUCTURE
            )
            # Spine skeletons
            np.testing.assert_array_equal(
                h5[f"{GRP_SPINES}/{GRP_SKELETONS}/neuron_A/points"][:], SAMPLE_POINTS
            )
            # Soma mesh
            np.testing.assert_array_equal(
                h5[f"{GRP_SOMA}/{GRP_MESHES}/neuron_A/vertices"][:], SAMPLE_VERTICES
            )
            # Spine mesh
            np.testing.assert_array_equal(
                h5[f"{GRP_SPINES}/{GRP_MESHES}/neuron_A/vertices"][:], SAMPLE_VERTICES
            )

    def test_multi_neuron_single_file(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A", n_spines=3)
        write_morphology(str(src), "neuron_B", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        write_spine_table(str(src), "neuron_B", _make_spine_table("neuron_B", 4))
        write_spine_skeletons(str(src), "neuron_B", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output, include_meshes=False)

        with h5py.File(output, "r") as h5:
            assert set(h5[GRP_MORPH].keys()) == {"neuron_A", "neuron_B"}


class TestMergeWithRename:
    def test_renames_all_groups(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "old")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output, rename_map={(src, "old"): "new"})

        with h5py.File(output, "r") as h5:
            for grp in [
                GRP_MORPH,
                GRP_EDGES,
                f"{GRP_SOMA}/{GRP_MESHES}",
                f"{GRP_SPINES}/{GRP_SKELETONS}",
                f"{GRP_SPINES}/{GRP_MESHES}",
            ]:
                assert "new" in h5[grp]
                assert "old" not in h5[grp]

    def test_updates_spine_morphology_column(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "old", n_spines=4)
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output, rename_map={(src, "old"): "new"})

        with h5py.File(output, "r") as h5:
            values = h5[f"{GRP_EDGES}/new/{COL_SPINE_MORPH}"][:]
            decoded = {v.decode() if isinstance(v, bytes) else str(v) for v in values}
            assert decoded == {"new"}

    def test_shared_spine_group_not_renamed(self, tmp_path):
        """Shared spine group keeps its name when only the neuron is renamed."""
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A", spine_group_name="shared_lib")
        output = tmp_path / "out.h5"

        # Two rename entries for the same file: neuron renamed, lib keeps its name
        merge_morphologies_with_spines(
            [src],
            output,
            rename_map={(src, "neuron_A"): "renamed", (src, "shared_lib"): "shared_lib"},
        )

        with h5py.File(output, "r") as h5:
            assert "renamed" in h5[GRP_MORPH]
            assert "shared_lib" in h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
            assert "renamed" not in h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]

    def test_shared_spine_group_overlap_raises(self, tmp_path):
        """Two files with the same shared spine group name without rename -> rejected."""
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A", spine_group_name="shared")
        _create_source_file(src2, "neuron_B", spine_group_name="shared")

        with pytest.raises(ValueError, match="Duplicate spines library destination name"):
            merge_morphologies_with_spines([src1, src2], tmp_path / "out.h5")

    def test_shared_spine_group_overlap_with_rename(self, tmp_path):
        """Two files with same shared spine group, renamed to different names -> OK."""
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A", spine_group_name="shared")
        _create_source_file(src2, "neuron_B", spine_group_name="shared")
        output = tmp_path / "out.h5"

        rename = {(src1, "shared"): "shared_from_1", (src2, "shared"): "shared_from_2"}
        merge_morphologies_with_spines([src1, src2], output, rename_map=rename)

        with h5py.File(output, "r") as h5:
            assert "shared_from_1" in h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
            assert "shared_from_2" in h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
            # spine_morphology columns updated
            vals_a = h5[f"{GRP_EDGES}/neuron_A/{COL_SPINE_MORPH}"][:]
            assert all(
                (v.decode() if isinstance(v, bytes) else str(v)) == "shared_from_1" for v in vals_a
            )
            vals_b = h5[f"{GRP_EDGES}/neuron_B/{COL_SPINE_MORPH}"][:]
            assert all(
                (v.decode() if isinstance(v, bytes) else str(v)) == "shared_from_2" for v in vals_b
            )

    def test_rename_only_specified_keys(self, tmp_path):
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A")
        _create_source_file(src2, "neuron_B")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines(
            [src1, src2], output, rename_map={(src1, "neuron_A"): "renamed_A"}
        )

        with h5py.File(output, "r") as h5:
            assert set(h5[GRP_MORPH].keys()) == {"renamed_A", "neuron_B"}

    def test_duplicate_names_across_files_with_rename(self, tmp_path):
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A")
        _create_source_file(src2, "neuron_A")
        output = tmp_path / "out.h5"

        rename = {(src1, "neuron_A"): "from_file1", (src2, "neuron_A"): "from_file2"}
        merge_morphologies_with_spines([src1, src2], output, rename_map=rename)

        with h5py.File(output, "r") as h5:
            assert set(h5[GRP_MORPH].keys()) == {"from_file1", "from_file2"}

    def test_longer_name_not_truncated(self, tmp_path):
        """Regression: renaming to a longer string must not truncate."""
        src = tmp_path / "src.h5"
        _create_source_file(src, "a", n_spines=3)
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines(
            [src], output, rename_map={(src, "a"): "very_long_destination_name"}
        )

        with h5py.File(output, "r") as h5:
            values = h5[f"{GRP_EDGES}/very_long_destination_name/{COL_SPINE_MORPH}"][:]
            decoded = [v.decode() if isinstance(v, bytes) else str(v) for v in values]
            assert all(v == "very_long_destination_name" for v in decoded)


class TestIncludeMeshes:
    def test_meshes_included_by_default(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output)

        with h5py.File(output, "r") as h5:
            assert f"{GRP_SOMA}/{GRP_MESHES}/neuron_A" in h5
            assert f"{GRP_SPINES}/{GRP_MESHES}/neuron_A" in h5

    def test_meshes_excluded(self, tmp_path):
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output, include_meshes=False)

        with h5py.File(output, "r") as h5:
            assert f"{GRP_SOMA}/{GRP_MESHES}" not in h5
            assert f"{GRP_SPINES}/{GRP_MESHES}" not in h5
            # Skeletons always copied
            assert f"{GRP_SPINES}/{GRP_SKELETONS}/neuron_A" in h5

    def test_missing_soma_meshes_in_source(self, tmp_path):
        """Source without soma meshes merges fine with include_meshes=True."""
        src = tmp_path / "src.h5"
        write_morphology(str(src), "neuron_A", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        write_spine_table(str(src), "neuron_A", _make_spine_table("neuron_A", 2))
        write_spine_skeletons(str(src), "neuron_A", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output, include_meshes=True)

        with h5py.File(output, "r") as h5:
            assert "neuron_A" in h5[GRP_MORPH]
            assert f"{GRP_SOMA}/{GRP_MESHES}" not in h5


class TestEdgeCases:
    def test_missing_spine_skeletons_group_raises(self, tmp_path):
        """Source with spines table but no /spines/skeletons is rejected."""
        src = tmp_path / "src.h5"
        write_morphology(str(src), "neuron_A", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        write_spine_table(str(src), "neuron_A", _make_spine_table("neuron_A", 2))

        with pytest.raises(ValueError, match="Invalid file: No /spines/skeletons group"):
            merge_morphologies_with_spines([src], tmp_path / "out.h5", include_meshes=False)

    def test_missing_referenced_spine_group_raises(self, tmp_path):
        """Source file referencing a spine group not in /spines/skeletons is rejected."""
        src = tmp_path / "src.h5"
        write_morphology(str(src), "neuron_A", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        write_spine_table(str(src), "neuron_A", _make_spine_table("missing_group", 2))
        write_spine_skeletons(str(src), "other_group", SAMPLE_POINTS, SAMPLE_STRUCTURE)

        with pytest.raises(ValueError, match="not found in /spines/skeletons"):
            merge_morphologies_with_spines([src], tmp_path / "out.h5")

    def test_referential_integrity_after_rename(self, tmp_path):
        """All spine_morphology values reference existing spines groups."""
        src1 = tmp_path / "src1.h5"
        src2 = tmp_path / "src2.h5"
        _create_source_file(src1, "neuron_A")
        _create_source_file(src2, "neuron_B")
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines(
            [src1, src2], output, rename_map={(src1, "neuron_A"): "renamed_A"}
        )

        with h5py.File(output, "r") as h5:
            skeleton_keys = set(h5[f"{GRP_SPINES}/{GRP_SKELETONS}"].keys())
            for neuron_key in h5[GRP_EDGES].keys():
                values = h5[f"{GRP_EDGES}/{neuron_key}/{COL_SPINE_MORPH}"][:]
                decoded = {v.decode() if isinstance(v, bytes) else str(v) for v in values}
                assert decoded.issubset(skeleton_keys)

    def test_shared_spine_group_in_same_file_copied_once(self, tmp_path):
        """Two morphologies in the same file sharing a spine group -> copied once."""
        src = tmp_path / "src.h5"
        _create_source_file(src, "neuron_A", spine_group_name="shared")
        # Add second neuron referencing the same shared group
        write_morphology(str(src), "neuron_B", SAMPLE_POINTS, SAMPLE_STRUCTURE)
        write_spine_table(str(src), "neuron_B", _make_spine_table("shared", 2))
        output = tmp_path / "out.h5"

        merge_morphologies_with_spines([src], output, include_meshes=False)

        with h5py.File(output, "r") as h5:
            assert set(h5[GRP_MORPH].keys()) == {"neuron_A", "neuron_B"}
            assert "shared" in h5[f"{GRP_SPINES}/{GRP_SKELETONS}"]
