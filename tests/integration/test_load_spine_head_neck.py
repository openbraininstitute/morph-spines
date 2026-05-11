"""Integration tests for head/neck triangle classification loading and filtering."""

import numpy as np
import pytest

from morph_spines.utils.morph_spine_loader import load_morphology_with_spines


@pytest.fixture
def morph_with_head_neck(single_morph_spines_head_neck_file):
    """Load the morphology with head/neck data, meshes preloaded."""
    return load_morphology_with_spines(
        str(single_morph_spines_head_neck_file),
        spines_are_centered=True,
        load_meshes=True,
    )


@pytest.fixture
def morph_with_head_neck_lazy(single_morph_spines_head_neck_file):
    """Load the morphology with head/neck data, meshes NOT preloaded (lazy)."""
    return load_morphology_with_spines(
        str(single_morph_spines_head_neck_file),
        spines_are_centered=True,
        load_meshes=False,
    )


class TestHeadNeckPreloaded:
    """Tests with preloaded meshes (load_meshes=True)."""

    def test_spine_count(self, morph_with_head_neck):
        assert morph_with_head_neck.spines.spine_count == 4

    def test_full_mesh_has_all_triangles(self, morph_with_head_neck):
        """Default spine_mesh returns all 4 triangles (tetrahedron)."""
        mesh = morph_with_head_neck.spines.spine_mesh(0)
        assert len(mesh.faces) == 4

    def test_neck_only_mesh(self, morph_with_head_neck):
        """include_head=False returns only neck triangles (3 for tetrahedron)."""
        mesh = morph_with_head_neck.spines.spine_mesh(0, include_head=False)
        assert len(mesh.faces) == 3

    def test_head_only_mesh(self, morph_with_head_neck):
        """include_neck=False returns only head triangles (1 for tetrahedron)."""
        mesh = morph_with_head_neck.spines.spine_mesh(0, include_head=True, include_neck=False)
        assert len(mesh.faces) == 1

    def test_neck_vertices_are_subset(self, morph_with_head_neck):
        """Neck mesh vertices are a subset of the full mesh vertices."""
        full_mesh = morph_with_head_neck.spines.spine_mesh(0)
        neck_mesh = morph_with_head_neck.spines.spine_mesh(0, include_head=False)

        # All neck vertices should exist in the full mesh
        for v in neck_mesh.vertices:
            assert any(np.allclose(v, fv) for fv in full_mesh.vertices)

    def test_head_vertices_are_subset(self, morph_with_head_neck):
        """Head mesh vertices are a subset of the full mesh vertices."""
        full_mesh = morph_with_head_neck.spines.spine_mesh(0)
        head_mesh = morph_with_head_neck.spines.spine_mesh(0, include_neck=False)

        for v in head_mesh.vertices:
            assert any(np.allclose(v, fv) for fv in full_mesh.vertices)

    def test_all_spines_have_head_neck(self, morph_with_head_neck):
        """All 4 spines have the same head/neck split (tetrahedron shape)."""
        for i in range(4):
            neck_mesh = morph_with_head_neck.spines.spine_mesh(i, include_head=False)
            head_mesh = morph_with_head_neck.spines.spine_mesh(i, include_neck=False)
            assert len(neck_mesh.faces) == 3
            assert len(head_mesh.faces) == 1

    def test_centered_spine_mesh_filtering(self, morph_with_head_neck):
        """centered_spine_mesh also supports head/neck filtering."""
        mesh = morph_with_head_neck.spines.centered_spine_mesh(0, include_head=False)
        assert len(mesh.faces) == 3

    def test_both_false_raises(self, morph_with_head_neck):
        """Both False raises ValueError."""
        with pytest.raises(ValueError):
            morph_with_head_neck.spines.spine_mesh(0, include_head=False, include_neck=False)


class TestHeadNeckLazy:
    """Tests with lazy mesh loading (load_meshes=False)."""

    def test_full_mesh_has_all_triangles(self, morph_with_head_neck_lazy):
        """Default spine_mesh returns all 4 triangles."""
        mesh = morph_with_head_neck_lazy.spines.spine_mesh(0)
        assert len(mesh.faces) == 4

    def test_neck_only_mesh(self, morph_with_head_neck_lazy):
        """include_head=False returns only neck triangles."""
        mesh = morph_with_head_neck_lazy.spines.spine_mesh(0, include_head=False)
        assert len(mesh.faces) == 3

    def test_head_only_mesh(self, morph_with_head_neck_lazy):
        """include_neck=False returns only head triangles."""
        mesh = morph_with_head_neck_lazy.spines.spine_mesh(0, include_neck=False)
        assert len(mesh.faces) == 1


class TestOldFormatBackwardCompat:
    """Tests that old files without head/neck data still work correctly."""

    def test_old_file_full_mesh(self, single_morph_spines_centered_file):
        """Old file: spine_mesh returns all triangles."""
        m = load_morphology_with_spines(
            str(single_morph_spines_centered_file),
            spines_are_centered=True,
            load_meshes=True,
        )
        mesh = m.spines.spine_mesh(0)
        assert len(mesh.faces) == 4

    def test_old_file_filtering_returns_all(self, single_morph_spines_centered_file):
        """Old file: filtering has no effect (all triangles are undefined)."""
        m = load_morphology_with_spines(
            str(single_morph_spines_centered_file),
            spines_are_centered=True,
            load_meshes=True,
        )
        # With no head/neck data, include_head=False should still return all triangles
        # because all are "undefined" and undefined is always returned
        neck_mesh = m.spines.spine_mesh(0, include_head=False)
        full_mesh = m.spines.spine_mesh(0)
        assert len(neck_mesh.faces) == len(full_mesh.faces)
