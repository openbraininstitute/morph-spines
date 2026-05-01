"""Tests for head/neck triangle filtering in spine meshes."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from morph_spines.utils.mesh import filter_triangles_by_head_neck, submesh


@pytest.fixture
def triangles():
    """10 triangles: 4 neck, 6 head."""
    return np.array(
        [
            [0, 1, 2],  # 0 - neck
            [0, 2, 3],  # 1 - neck
            [0, 3, 1],  # 2 - neck
            [1, 3, 2],  # 3 - neck
            [4, 5, 6],  # 4 - head
            [4, 6, 7],  # 5 - head
            [4, 7, 5],  # 6 - head
            [5, 7, 6],  # 7 - head
            [8, 9, 10],  # 8 - head
            [8, 10, 11],  # 9 - head
        ],
        dtype=int,
    )


class TestFilterTrianglesByHeadNeck:
    """Tests for the _filter_triangles_by_head_neck helper.

    Offset convention (H heads):
      offsets has H+1 entries: [head_0_start, head_1_start, ..., total_triangles]
      - Neck:   triangles[0 : offsets[0]]
      - Head N: triangles[offsets[N] : offsets[N+1]]

    Return value is a list[NDArray] ordered as [neck, head_0, head_1, ...],
    filtered by include_head / include_neck.
    """

    def test_empty_offsets_returns_all(self, triangles):
        """No offsets means all triangles are undefined and always returned."""
        result = filter_triangles_by_head_neck(
            triangles,
            np.array([], dtype=int),
            include_head=True,
            include_neck=True,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_empty_offsets_include_head_only(self, triangles):
        """No offsets: undefined triangles returned even when include_neck=False."""
        result = filter_triangles_by_head_neck(
            triangles,
            np.array([], dtype=int),
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_empty_offsets_include_neck_only(self, triangles):
        """No offsets: undefined triangles returned even when include_head=False."""
        result = filter_triangles_by_head_neck(
            triangles,
            np.array([], dtype=int),
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_include_both(self, triangles):
        """With offsets, both True returns [neck, head_0]."""
        offsets = np.array([4, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=True,
        )
        assert len(result) == 2
        assert_array_equal(result[0], triangles[:4])  # neck
        assert_array_equal(result[1], triangles[4:])  # head 0

    def test_include_neck_only(self, triangles):
        """include_head=False returns [neck] only."""
        offsets = np.array([4, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles[:4])

    def test_include_head_only(self, triangles):
        """include_neck=False returns [head_0] only."""
        offsets = np.array([4, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles[4:])

    def test_all_head_no_neck(self, triangles):
        """offsets[0]==0 means neck is empty; neck-only returns empty list."""
        offsets = np.array([0, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 1
        assert len(result[0]) == 0

    def test_all_head_no_neck_head_only(self, triangles):
        """offsets[0]==0, requesting head returns [head_0] covering all."""
        offsets = np.array([0, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_all_neck_no_head(self, triangles):
        """offsets[0]==total means head is empty; head-only returns empty list."""
        n = len(triangles)
        offsets = np.array([n, n], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 1
        assert len(result[0]) == 0

    def test_all_neck_no_head_neck_only(self, triangles):
        """offsets[0]==total, requesting neck returns [neck] covering all."""
        n = len(triangles)
        offsets = np.array([n, n], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_branched_spine_two_heads_neck_only(self, triangles):
        """Branched: neck=[0..2), head_0=[2..6), head_1=[6..10). Neck only."""
        offsets = np.array([2, 6, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles[:2])

    def test_branched_spine_head_only(self, triangles):
        """Branched: head-only returns [head_0, head_1]."""
        offsets = np.array([2, 6, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 2
        assert_array_equal(result[0], triangles[2:6])
        assert_array_equal(result[1], triangles[6:10])

    def test_branched_spine_both(self, triangles):
        """Branched: both returns [neck, head_0, head_1]."""
        offsets = np.array([2, 6, 10], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=True,
        )
        assert len(result) == 3
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[2:6])
        assert_array_equal(result[2], triangles[6:10])

    def test_single_head_access_pattern(self, triangles):
        """Verify offsets[N]:offsets[N+1] access pattern for head N."""
        offsets = np.array([4, 10], dtype=int)
        head_0 = triangles[offsets[0] : offsets[1]]
        assert len(head_0) == 6
        assert_array_equal(head_0, triangles[4:])

    def test_branched_head_access_pattern(self, triangles):
        """Verify offsets[N]:offsets[N+1] for each head in a branched spine."""
        offsets = np.array([2, 6, 10], dtype=int)
        head_0 = triangles[offsets[0] : offsets[1]]
        assert len(head_0) == 4
        head_1 = triangles[offsets[1] : offsets[2]]
        assert len(head_1) == 4
        neck = triangles[: offsets[0]]
        assert len(neck) == 2


class TestSubmesh:
    """Tests for the _submesh helper."""

    def test_submesh_filters_vertices(self):
        """Only vertices referenced by triangles are kept."""
        vertices = np.array(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [1, 1, 0],  # not referenced
                [0, 0, 1],
            ],
            dtype=float,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 4]], dtype=int)

        new_verts, new_tris = submesh(vertices, triangles)

        assert len(new_verts) == 4  # indices 0, 1, 2, 4
        assert_array_equal(new_verts, vertices[[0, 1, 2, 4]])
        # Remapped: 0->0, 1->1, 2->2, 4->3
        assert_array_equal(new_tris, [[0, 1, 2], [0, 2, 3]])

    def test_submesh_empty_triangles(self):
        """Empty triangles produce empty output."""
        vertices = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        triangles = np.empty((0, 3), dtype=int)

        new_verts, new_tris = submesh(vertices, triangles)

        assert len(new_verts) == 0
        assert len(new_tris) == 0

    def test_submesh_all_referenced(self):
        """When all vertices are referenced, output matches input."""
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
        triangles = np.array([[0, 1, 2]], dtype=int)

        new_verts, new_tris = submesh(vertices, triangles)

        assert_array_equal(new_verts, vertices)
        assert_array_equal(new_tris, triangles)


class TestSpinesMeshHeadNeck:
    """Tests for head/neck filtering through the Spines class mesh methods."""

    def test_spine_mesh_both_false_raises(self, spines_with_meshes):
        with pytest.raises(ValueError, match="At least one of"):
            spines_with_meshes.spine_mesh(
                0,
                include_head=False,
                include_neck=False,
            )

    def test_centered_spine_mesh_both_false_raises(self, spines_with_meshes):
        with pytest.raises(ValueError, match="At least one of"):
            spines_with_meshes.centered_spine_mesh(
                0,
                include_head=False,
                include_neck=False,
            )

    def test_spine_mesh_triangles_returns_flat_array(
        self,
        spines_with_meshes,
        spines_meshes,
    ):
        """spine_mesh_triangles returns a flat NDArray (no filtering)."""
        spine_id = 0
        expected = spines_meshes[spine_id].faces
        result = spines_with_meshes.spine_mesh_triangles(spine_id)
        assert isinstance(result, np.ndarray)
        assert_array_equal(result, expected)


# Fixtures for TestSpinesMeshHeadNeck

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
            PointLevel([spine_start, spine_end], [1, 1]),
            SectionType.axon,
        )
    return Morphology(
        spines.as_immutable(),
        spines_collection,
        process_subtrees=False,
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
