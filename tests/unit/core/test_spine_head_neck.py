"""Tests for head/neck classification in the Spines class."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal


class TestSpinesMeshHeadNeck:
    """Tests for head/neck filtering through the Spines class mesh methods."""

    def test_spine_mesh_both_false_raises(self, spines_with_meshes):
        with pytest.raises(ValueError, match="At least one of"):
            spines_with_meshes.spine_mesh(
                0, include_head=False, include_neck=False,
            )

    def test_centered_spine_mesh_both_false_raises(self, spines_with_meshes):
        with pytest.raises(ValueError, match="At least one of"):
            spines_with_meshes.centered_spine_mesh(
                0, include_head=False, include_neck=False,
            )

    def test_spine_mesh_triangles_returns_flat_array(
        self, spines_with_meshes, spines_meshes,
    ):
        """spine_mesh_triangles returns a flat NDArray (no filtering)."""
        spine_id = 0
        expected = spines_meshes[spine_id].faces
        result = spines_with_meshes.spine_mesh_triangles(spine_id)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, expected)


class TestSpinesConstructorWarnings:
    """Tests for warning messages when constructing Spines with invalid combinations."""

    def test_meshes_without_offsets_warns(
        self, capsys, spines_table, spines_skeletons, spines_meshes,
    ):
        """Providing spine_meshes without head_neck_offsets prints a warning."""
        from morph_spines.core.spines import Spines

        s = Spines(
            meshes_filepath="spines.h5",
            morphology_name="collection_0",
            spine_table=spines_table,
            centered_spine_skeletons=spines_skeletons,
            spines_are_centered=False,
            spine_meshes=spines_meshes,
            head_neck_offsets=None,
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "head_neck_offsets" in captured.out
        assert s._valid_head_neck_offsets is False

    def test_offsets_without_meshes_warns(
        self, capsys, spines_table, spines_skeletons,
    ):
        """Providing head_neck_offsets without spine_meshes prints a warning."""
        from morph_spines.core.spines import Spines

        offsets = [np.array([0, 2, 4], dtype=int)] * 4

        s = Spines(
            meshes_filepath="spines.h5",
            morphology_name="collection_0",
            spine_table=spines_table,
            centered_spine_skeletons=spines_skeletons,
            spines_are_centered=False,
            spine_meshes=None,
            head_neck_offsets=offsets,
        )

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "ignored" in captured.out
        assert s._head_neck_offsets == []


class TestGetHeadNeckOffsetsLazy:
    """Tests for _get_head_neck_offsets reading from H5 (lazy path)."""

    def test_undefined_spine_in_file_with_head_neck(self, tmp_path):
        """A spine with equal hn offsets (hn_start == hn_end) returns empty."""
        import h5py
        import pandas as pd
        from morphio import PointLevel, SectionType
        from neurom.core.morphology import Morphology
        from scipy.spatial.transform import Rotation

        from morph_spines.core.h5_schema import (
            COL_AFF_SEC,
            COL_ROTATION,
            COL_SPINE_ID,
            COL_SPINE_MORPH,
            COL_TRANSLATION,
        )
        from morph_spines.core.spines import Spines

        # Create a minimal H5 file with 2 spines: spine 0 has head/neck, spine 1 is undefined
        filepath = str(tmp_path / "test.h5")
        with h5py.File(filepath, "w") as f:
            grp = f.create_group("spines/meshes/coll_0")
            vertices = np.zeros((8, 3), dtype=float)
            triangles = np.zeros((8, 3), dtype=int)
            # Spine 0: hn values at [0:3], spine 1: hn values at [3:3] (empty = undefined)
            offsets = np.array([
                [0, 0, 0],
                [4, 4, 3],
                [8, 8, 3],
            ], dtype=int)
            head_neck_values = np.array([0, 3, 4], dtype=int)

            grp.create_dataset("vertices", data=vertices)
            grp.create_dataset("triangles", data=triangles)
            grp.create_dataset("offsets", data=offsets)
            grp.create_dataset("head_neck_values", data=head_neck_values)

        num_spines = 2
        rotation = np.tile(Rotation.identity().as_quat(), (num_spines, 1))
        translation = np.zeros((num_spines, 3), dtype=float)
        spine_table = pd.DataFrame({
            COL_SPINE_ID: [0, 1],
            COL_SPINE_MORPH: ["coll_0", "coll_0"],
            COL_ROTATION[0]: rotation[:, 0],
            COL_ROTATION[1]: rotation[:, 1],
            COL_ROTATION[2]: rotation[:, 2],
            COL_ROTATION[3]: rotation[:, 3],
            COL_TRANSLATION[0]: translation[:, 0],
            COL_TRANSLATION[1]: translation[:, 1],
            COL_TRANSLATION[2]: translation[:, 2],
            COL_AFF_SEC: [0, 0],
        })

        import morphio

        skeletons = morphio.mut.Morphology()
        for _ in range(num_spines):
            skeletons.append_root_section(
                PointLevel([[0, 0, 0], [0, 1, 0]], [1, 1]), SectionType.axon,
            )
        centered_skeletons = Morphology(
            skeletons.as_immutable(), "coll_0", process_subtrees=False,
        )

        s = Spines(
            meshes_filepath=filepath,
            morphology_name="coll_0",
            spine_table=spine_table,
            centered_spine_skeletons=centered_skeletons,
            spines_are_centered=False,
            spine_meshes=None,
            head_neck_offsets=None,
        )

        # Spine 0 has head/neck data
        offsets_0 = s._get_head_neck_offsets(0)
        assert_array_equal(offsets_0, [0, 3, 4])

        # Spine 1 is undefined (hn_start == hn_end)
        offsets_1 = s._get_head_neck_offsets(1)
        assert len(offsets_1) == 0


class TestSpineType:
    """Tests for the spine_type method."""

    def test_spine_type_missing_column(self, spines_with_meshes):
        """spine_type returns UNDEFINED when the column is not in the spine table."""
        from morph_spines.core.spine_type import SpineType

        assert spines_with_meshes.spine_type(0) == SpineType.UNDEFINED

    def test_spine_type_with_column(self, spines_with_meshes):
        """spine_type returns the correct SpineType when the column is present."""
        from morph_spines.core.h5_schema import COL_SPINE_TYPE
        from morph_spines.core.spine_type import SpineType

        spines_with_meshes.spine_table[COL_SPINE_TYPE] = "mushroom"
        assert spines_with_meshes.spine_type(0) == SpineType.MUSHROOM

        spines_with_meshes.spine_table.loc[1, COL_SPINE_TYPE] = "thin"
        assert spines_with_meshes.spine_type(1) == SpineType.THIN


# --- Fixtures ---


@pytest.fixture
def num_spines():
    return 4


@pytest.fixture
def spines_collection():
    return "collection_0"


@pytest.fixture
def spines_table(num_spines, spines_collection):
    import pandas as pd
    from scipy.spatial.transform import Rotation

    from morph_spines.core.h5_schema import (
        COL_AFF_SEC,
        COL_ROTATION,
        COL_SPINE_ID,
        COL_SPINE_MORPH,
        COL_TRANSLATION,
    )

    rotation = np.tile(Rotation.identity().as_quat(), (num_spines, 1))
    translation = np.tile([0.0, 0.0, 0.0], (num_spines, 1)).astype(float)
    return pd.DataFrame(
        {
            COL_SPINE_ID: range(num_spines),
            COL_SPINE_MORPH: [spines_collection] * num_spines,
            COL_ROTATION[0]: rotation[:, 0],
            COL_ROTATION[1]: rotation[:, 1],
            COL_ROTATION[2]: rotation[:, 2],
            COL_ROTATION[3]: rotation[:, 3],
            COL_TRANSLATION[0]: translation[:, 0],
            COL_TRANSLATION[1]: translation[:, 1],
            COL_TRANSLATION[2]: translation[:, 2],
            COL_AFF_SEC: [int(2 + i / 2) for i in range(num_spines)],
        }
    )


@pytest.fixture
def spines_skeletons(num_spines, spines_collection):
    import morphio
    from morphio import PointLevel, SectionType
    from neurom.core.morphology import Morphology

    spines = morphio.mut.Morphology()
    for idx in range(num_spines):
        spine_start = [float(idx), 0.0, 0.0]
        spine_end = [float(idx) + 1.0, 0.0, 0.0]
        spines.append_root_section(
            PointLevel([spine_start, spine_end], [1, 1]), SectionType.axon,
        )
    return Morphology(
        spines.as_immutable(), spines_collection, process_subtrees=False,
    )


@pytest.fixture
def spines_meshes(spines_skeletons):
    import trimesh

    meshes = []
    for section in spines_skeletons.to_morphio().root_sections:
        vertices = []
        spine_start = section.points[0]
        vertices.append(spine_start)
        end_x, end_y, end_z = section.points[-1]
        vertices.append([end_x + 0.5, end_y, end_z])
        vertices.append([end_x, end_y + 0.5, end_z])
        vertices.append([end_x, end_y, end_z + 0.5])
        tri = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]]
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=tri))
    return meshes


@pytest.fixture
def spines_with_meshes(spines_table, spines_skeletons, spines_meshes):
    from morph_spines.core.spines import Spines

    return Spines(
        meshes_filepath="spines.h5",
        morphology_name="collection_0",
        spine_table=spines_table,
        centered_spine_skeletons=spines_skeletons,
        spines_are_centered=False,
        spine_meshes=spines_meshes,
        head_neck_offsets=[np.array([], dtype=int)] * len(spines_meshes),
    )
