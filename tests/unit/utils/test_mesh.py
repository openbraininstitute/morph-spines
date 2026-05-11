"""Tests for morph_spines.utils.mesh utilities."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from morph_spines.utils.mesh import filter_triangles_by_head_neck, submesh


@pytest.fixture
def triangles():
    """12 triangles: 2 undefined, 4 neck, 6 head."""
    return np.array(
        [
            [0, 1, 2],  # 0 - undefined
            [0, 2, 3],  # 1 - undefined
            [0, 3, 1],  # 2 - neck
            [1, 3, 2],  # 3 - neck
            [2, 3, 0],  # 4 - neck
            [3, 2, 1],  # 5 - neck
            [4, 5, 6],  # 6 - head
            [4, 6, 7],  # 7 - head
            [4, 7, 5],  # 8 - head
            [5, 7, 6],  # 9 - head
            [8, 9, 10],  # 10 - head
            [8, 10, 11],  # 11 - head
        ],
        dtype=int,
    )


class TestFilterTrianglesByHeadNeck:
    """Tests for filter_triangles_by_head_neck.

    Offset convention (H heads):
      offsets has H+2 entries: [undefined_end, neck_end, head_0_end, ..., total]
      - Undefined: triangles[0 : offsets[0]]             (always included)
      - Neck:      triangles[offsets[0] : offsets[1]]
      - Head N:    triangles[offsets[N+1] : offsets[N+2]]

    Return value is a list[NDArray] ordered as:
      [undefined, neck, head_0, head_1, ...],
    filtered by include_head / include_neck. Undefined is always present.
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
        """No offsets: all returned even when include_neck=False."""
        result = filter_triangles_by_head_neck(
            triangles,
            np.array([], dtype=int),
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_empty_offsets_include_neck_only(self, triangles):
        """No offsets: all returned even when include_head=False."""
        result = filter_triangles_by_head_neck(
            triangles,
            np.array([], dtype=int),
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles)

    def test_include_both(self, triangles):
        """Both True returns [undefined, neck, head_0]."""
        offsets = np.array([2, 6, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=True,
        )
        assert len(result) == 3
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[2:6])
        assert_array_equal(result[2], triangles[6:])

    def test_include_neck_only(self, triangles):
        """include_head=False returns [undefined, neck]."""
        offsets = np.array([2, 6, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 2
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[2:6])

    def test_include_head_only(self, triangles):
        """include_neck=False returns [undefined, head_0]."""
        offsets = np.array([2, 6, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 2
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[6:])

    def test_no_undefined_include_both(self, triangles):
        """No undefined triangles (offsets[0]==0), both True."""
        offsets = np.array([0, 4, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=True,
        )
        assert len(result) == 3
        assert len(result[0]) == 0
        assert_array_equal(result[1], triangles[:4])
        assert_array_equal(result[2], triangles[4:])

    def test_no_undefined_neck_only(self, triangles):
        """No undefined, neck only."""
        offsets = np.array([0, 4, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 2
        assert len(result[0]) == 0
        assert_array_equal(result[1], triangles[:4])

    def test_no_undefined_head_only(self, triangles):
        """No undefined, head only."""
        offsets = np.array([0, 4, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 2
        assert len(result[0]) == 0
        assert_array_equal(result[1], triangles[4:])

    def test_all_head_no_neck(self, triangles):
        """Neck is empty (offsets[0]==offsets[1]), neck-only returns [undefined, empty neck]."""
        offsets = np.array([0, 0, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == 0

    def test_all_neck_no_head(self, triangles):
        """Head is empty (offsets[1]==offsets[2]==total), head-only returns [undefined, empty]."""
        n = len(triangles)
        offsets = np.array([0, n, n], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 2
        assert len(result[0]) == 0
        assert len(result[1]) == 0

    def test_branched_spine_two_heads_neck_only(self, triangles):
        """Branched: undefined=[0..2), neck=[2..4), head_0=[4..8), head_1=[8..12)."""
        offsets = np.array([2, 4, 8, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=True,
        )
        assert len(result) == 2
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[2:4])

    def test_branched_spine_head_only(self, triangles):
        """Branched: head-only returns [undefined, head_0, head_1]."""
        offsets = np.array([2, 4, 8, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=False,
        )
        assert len(result) == 3
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[4:8])
        assert_array_equal(result[2], triangles[8:12])

    def test_branched_spine_both(self, triangles):
        """Branched: both returns [undefined, neck, head_0, head_1]."""
        offsets = np.array([2, 4, 8, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=True,
            include_neck=True,
        )
        assert len(result) == 4
        assert_array_equal(result[0], triangles[:2])
        assert_array_equal(result[1], triangles[2:4])
        assert_array_equal(result[2], triangles[4:8])
        assert_array_equal(result[3], triangles[8:12])

    def test_neither_head_nor_neck(self, triangles):
        """Both False returns only [undefined]."""
        offsets = np.array([2, 6, 12], dtype=int)
        result = filter_triangles_by_head_neck(
            triangles,
            offsets,
            include_head=False,
            include_neck=False,
        )
        assert len(result) == 1
        assert_array_equal(result[0], triangles[:2])


class TestSubmesh:
    """Tests for the submesh helper."""

    def test_submesh_filters_vertices(self):
        """Only vertices referenced by triangles are kept."""
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1]],
            dtype=float,
        )
        triangles = np.array([[0, 1, 2], [0, 2, 4]], dtype=int)

        new_verts, new_tris = submesh(vertices, triangles)

        assert len(new_verts) == 4
        assert_array_equal(new_verts, vertices[[0, 1, 2, 4]])
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
