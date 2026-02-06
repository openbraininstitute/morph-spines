import numpy as np
import pandas as pd
import pytest
import trimesh
from morphio import PointLevel, SectionType
from morphio.mut import Morphology
from neurom.core.morphology import Morphology as neuromMorphology
from neurom.core.morphology import Neurite
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation

from morph_spines.core.h5_schema import (
    COL_AFF_SEC,
    COL_ROTATION,
    COL_SPINE_ID,
    COL_SPINE_MORPH,
    COL_TRANSLATION,
)
from morph_spines.core.spines import Spines
from morph_spines.utils import geometry


@pytest.fixture
def num_spines():
    return 4


@pytest.fixture
def spines_collection():
    return "collection_0"


@pytest.fixture
def spines_table(num_spines, spines_collection):
    rotation = np.tile(Rotation.identity().as_quat(), (num_spines, 1))
    translation = np.tile([0.0, 0.0, 0.0], (num_spines, 1)).astype(float)
    df = pd.DataFrame(
        {
            COL_SPINE_ID: range(num_spines),
            COL_SPINE_MORPH: [spines_collection] * num_spines,
            COL_ROTATION[0]: rotation[:, 0],  # X
            COL_ROTATION[1]: rotation[:, 1],  # Y
            COL_ROTATION[2]: rotation[:, 2],  # Z
            COL_ROTATION[3]: rotation[:, 3],  # W
            COL_TRANSLATION[0]: translation[:, 0],  # X
            COL_TRANSLATION[1]: translation[:, 1],  # Y
            COL_TRANSLATION[2]: translation[:, 2],  # Z
            COL_AFF_SEC: [int(2 + i / 2) for i in range(num_spines)],
        }
    )
    return df


@pytest.fixture
def centered_spines_table(spines_table):
    translation = np.array([[i, 0.0, 0.0] for i in range(len(spines_table))])
    centered_df = spines_table.copy()
    centered_df[list(COL_TRANSLATION)] = translation

    return centered_df


@pytest.fixture
def spines_skeletons(num_spines, spines_collection):
    spines = Morphology()

    for idx in range(num_spines):
        spine_start = [float(idx), 0.0, 0.0]
        spine_end = [float(idx) + 1.0, 0.0, 0.0]
        spines.append_root_section(PointLevel([spine_start, spine_end], [1, 1]), SectionType.axon)

    return neuromMorphology(spines.as_immutable(), spines_collection, process_subtrees=False)


@pytest.fixture
def spines_meshes(spines_skeletons):
    meshes = []
    for section in spines_skeletons.to_morphio().root_sections:
        vertices = []
        triangles = []
        spine_start = section.points[0]
        vertices.append(spine_start)  # spine mesh start: single point
        end_x, end_y, end_z = section.points[-1]  # spine mesh end: points will form a triangle
        vertices.append([end_x + 0.5, end_y, end_z])
        vertices.append([end_x, end_y + 0.5, end_z])
        vertices.append([end_x, end_y, end_z + 0.5])
        triangles.append([[0, 1, 2]])
        triangles.append([[0, 2, 3]])
        triangles.append([[0, 3, 1]])
        triangles.append([[1, 3, 2]])
        meshes.append(trimesh.Trimesh(vertices=vertices, faces=triangles))

    return meshes


@pytest.fixture
def centered_spines_meshes(spines_meshes, centered_spines_table):
    centered_meshes = []
    for idx, mesh in enumerate(spines_meshes):
        centered_spine_mesh = mesh.copy()
        spine_rotation = Rotation.from_quat(
            np.array(centered_spines_table.loc[idx, COL_ROTATION].to_numpy(dtype=float))
        )
        spine_translation = centered_spines_table.loc[idx, COL_TRANSLATION].to_numpy(dtype=float)
        transform_matrix = geometry.inverse_transform_matrix_for_spine(
            spine_rotation, spine_translation
        )
        centered_spine_mesh.apply_transform(transform_matrix)
        centered_meshes.append(centered_spine_mesh)

    return centered_meshes


@pytest.fixture
def spines_filename():
    return "spines.h5"


@pytest.fixture
def spines(spines_filename, spines_collection, spines_table, spines_skeletons):
    """Fixture providing a Spines instance -- with centered spines, without meshes"""
    return Spines(
        meshes_filepath=spines_filename,
        morphology_name=spines_collection,
        spine_table=spines_table,
        centered_spine_skeletons=spines_skeletons,
        spines_are_centered=False,
        spine_meshes=None,
    )


@pytest.fixture
def spines_with_meshes(spines, spines_meshes):
    spines_w_meshes = spines
    spines_w_meshes._spine_meshes = spines_meshes
    return spines_w_meshes


@pytest.fixture
def centered_spines(spines_filename, spines_collection, centered_spines_table, spines_skeletons):
    """Fixture providing a Spines instance -- with centered spines, without meshes"""
    return Spines(
        meshes_filepath=spines_filename,
        morphology_name=spines_collection,
        spine_table=centered_spines_table,
        centered_spine_skeletons=spines_skeletons,
        spines_are_centered=True,
        spine_meshes=None,
    )


@pytest.fixture
def centered_spines_with_meshes(centered_spines, spines_meshes):
    spines_w_meshes = centered_spines
    spines_w_meshes._spine_meshes = spines_meshes
    return spines_w_meshes


def test_spine_count(spines, num_spines):
    assert spines.spine_count == num_spines


def test_spine_transformations(centered_spines, spines_table):
    spine_id = 0
    expected_transformations = (
        np.eye(3),
        spines_table.loc[spine_id, COL_TRANSLATION].to_numpy(dtype=float),
    )
    transformations = centered_spines.spine_transformations(spine_id)

    assert len(transformations[0].as_matrix()) == len(expected_transformations[0])
    assert len(transformations[1]) == len(expected_transformations[1])

    assert np.allclose(transformations[0].as_matrix(), expected_transformations[0])
    assert np.allclose(transformations[1], expected_transformations[1])


def test__transform_spine_skeletons_fail_num_spines(centered_spines):
    centered_spines.spine_table.drop(index=0, inplace=True)

    with pytest.raises(ValueError):
        centered_spines._transform_spine_skeletons()


def test_spine_skeletons(spines, num_spines):
    spine_id = 0
    spine_skeletons = spines.spine_skeletons
    expected_points = np.array([[0, 0, 0, 0.5], [1, 0, 0, 0.5]], dtype=np.float32)

    assert len(spine_skeletons) == num_spines
    assert isinstance(spine_skeletons[spine_id], Neurite)
    assert spine_skeletons[spine_id].length == 1.0
    assert_allclose(spine_skeletons[spine_id].points, expected_points)


def test_centered_spine_skeletons(centered_spines, num_spines):
    spine_id = 0
    spine_skeletons = centered_spines.centered_spine_skeletons
    expected_points = np.array([[0, 0, 0, 0.5], [1, 0, 0, 0.5]], dtype=np.float32)

    assert len(spine_skeletons) == num_spines
    assert isinstance(spine_skeletons[spine_id], Neurite)
    assert spine_skeletons[spine_id].length == 1.0
    assert_allclose(spine_skeletons[spine_id].points, expected_points)


def test_spine_mesh_points(spines_with_meshes, spines_meshes):
    # Not centered spines, must return not centered mesh points (global coordinates)
    spine_id = 1
    expected_points = spines_meshes[spine_id].vertices
    points = spines_with_meshes.spine_mesh_points(spine_loc=spine_id)

    assert_array_equal(points, expected_points)


def test_spine_mesh_points_centered(centered_spines_with_meshes, spines_meshes):
    # Centered spines, must return not centered mesh points (global coordinates)
    spine_id = 1
    expected_points = spines_meshes[spine_id].vertices
    points = centered_spines_with_meshes.spine_mesh_points(spine_loc=spine_id)

    assert_array_equal(points, expected_points)


def test_centered_mesh_points(spines_with_meshes, spines_meshes):
    # Not centered spines, must return not centered mesh points (global coordinates),
    # since the spine table does not specify how to translate points to origin
    spine_id = 1
    expected_points = spines_meshes[spine_id].vertices
    points = spines_with_meshes.centered_mesh_points(spine_loc=spine_id)

    assert_array_equal(points, expected_points)


def test_centered_mesh_points_centered(centered_spines_with_meshes, centered_spines_meshes):
    # Centered spines, must return centered mesh points (local coordinates)
    spine_id = 1
    expected_points = centered_spines_meshes[spine_id].vertices
    points = centered_spines_with_meshes.centered_mesh_points(spine_loc=spine_id)

    assert_array_equal(points, expected_points)


def test_spine_mesh_triangles(spines_with_meshes, spines_meshes):
    spine_id = 0
    expected_triangles = spines_meshes[spine_id].faces
    triangles = spines_with_meshes.spine_mesh_triangles(spine_loc=spine_id)

    assert_array_equal(triangles, expected_triangles)


def test_spine_mesh(spines_with_meshes, spines_meshes):
    spine_id = 1
    mesh = spines_with_meshes.spine_mesh(spine_id)

    assert isinstance(mesh, trimesh.Trimesh)
    assert_array_equal(mesh.vertices, spines_meshes[spine_id].vertices)
    assert_array_equal(mesh.faces, spines_meshes[spine_id].faces)


def test_spine_mesh_centered(centered_spines_with_meshes, spines_meshes):
    spine_id = 1
    mesh = centered_spines_with_meshes.spine_mesh(spine_id)

    assert isinstance(mesh, trimesh.Trimesh)
    assert_array_equal(mesh.vertices, spines_meshes[spine_id].vertices)
    assert_array_equal(mesh.faces, spines_meshes[spine_id].faces)


def test_centered_spine_mesh(spines_with_meshes, spines_meshes):
    # Non-centered spines --> non-centered mesh
    spine_id = 1
    mesh = spines_with_meshes.centered_spine_mesh(spine_id)
    assert isinstance(mesh, trimesh.Trimesh)
    assert_array_equal(mesh.vertices, spines_meshes[spine_id].vertices)
    assert_array_equal(mesh.faces, spines_meshes[spine_id].faces)


def test_centered_spine_mesh_centered(centered_spines_with_meshes, centered_spines_meshes):
    spine_id = 1
    mesh = centered_spines_with_meshes.centered_spine_mesh(spine_id)
    assert isinstance(mesh, trimesh.Trimesh)
    assert_array_equal(mesh.vertices, centered_spines_meshes[spine_id].vertices)
    assert_array_equal(mesh.faces, centered_spines_meshes[spine_id].faces)


def test_spine_indices_for_section(spines, spines_table):
    sec_id = 2
    expected_indices = spines_table.loc[spines_table[COL_AFF_SEC] == sec_id].index.to_numpy()
    indices = spines.spine_indices_for_section(sec_id)

    assert_array_equal(expected_indices, indices)


def test_spine_table_for_section(spines, spines_table, spines_collection):
    sec_id = 2
    spine_table = spines.spine_table_for_section(sec_id)

    assert isinstance(spine_table, pd.DataFrame)
    assert len(spine_table.columns) == len(spines_table.columns)
    assert set(
        [
            COL_AFF_SEC,
            COL_SPINE_MORPH,
            COL_ROTATION[1],
            COL_TRANSLATION[0],
        ]
    ).issubset(set(spine_table.columns))
    assert spine_table.loc[0, COL_ROTATION[0]] == spines_table.loc[0, COL_ROTATION[0]]
    assert spine_table.loc[0, COL_SPINE_MORPH] == spines_collection


def test_spine_meshes_for_section(spines_with_meshes, spines_meshes, spines_table):
    sec_id = 2
    # Get the spine IDs for the given section
    spine_ids_for_sec = spines_table.loc[spines_table[COL_AFF_SEC] == sec_id][
        COL_SPINE_ID
    ].to_numpy()
    section_meshes = [spines_meshes[i] for i in spine_ids_for_sec]

    meshes = list(spines_with_meshes.spine_meshes_for_section(sec_id))

    assert len(meshes) == len(section_meshes)

    for mesh, expected_mesh in zip(meshes, section_meshes, strict=True):
        assert isinstance(mesh, trimesh.Trimesh)
        assert_allclose(mesh.vertices, expected_mesh.vertices)
        assert_allclose(mesh.faces, expected_mesh.faces)


def test_compound_spine_mesh_for_section(spines_with_meshes, spines_meshes, spines_table):
    sec_id = 2
    # Get the spine IDs for the given section
    spine_ids_for_sec = spines_table.loc[spines_table[COL_AFF_SEC] == sec_id][
        COL_SPINE_ID
    ].to_numpy()
    section_meshes = [spines_meshes[i] for i in spine_ids_for_sec]
    expected_mesh = trimesh.util.concatenate(section_meshes)

    mesh = spines_with_meshes.compound_spine_mesh_for_section(sec_id)

    assert isinstance(mesh, trimesh.Trimesh)
    assert_allclose(mesh.vertices, expected_mesh.vertices)
    assert_allclose(mesh.faces, expected_mesh.faces)


def test_centered_spine_meshes_for_section(
    centered_spines_with_meshes, centered_spines_meshes, spines_table
):
    sec_id = 2
    spine_ids_for_sec = spines_table.loc[spines_table[COL_AFF_SEC] == sec_id][
        COL_SPINE_ID
    ].to_numpy()
    section_meshes = [centered_spines_meshes[i] for i in spine_ids_for_sec]

    meshes = list(centered_spines_with_meshes.centered_spine_meshes_for_section(sec_id))

    assert len(meshes) == len(section_meshes)

    for mesh, expected_mesh in zip(meshes, section_meshes, strict=True):
        assert isinstance(mesh, trimesh.Trimesh)
        print(f"mesh: \n{mesh.vertices}, \nexpected_mesh: \n{expected_mesh.vertices}")
        assert_allclose(mesh.vertices, expected_mesh.vertices)
        assert_allclose(mesh.faces, expected_mesh.faces)


def test_spine_meshes_for_morphology(spines_with_meshes, spines_meshes):
    morph_meshes = list(spines_with_meshes.spine_meshes_for_morphology())

    assert len(morph_meshes) == len(spines_meshes)

    for mesh, expected_mesh in zip(morph_meshes, spines_meshes, strict=True):
        assert isinstance(mesh, trimesh.Trimesh)
        print(f"mesh: \n{mesh.vertices}, \nexpected_mesh: \n{expected_mesh.vertices}")
        assert_allclose(mesh.vertices, expected_mesh.vertices)
        assert_allclose(mesh.faces, expected_mesh.faces)


def test_compound_spine_meshes_for_morphology(spines_with_meshes, spines_meshes):
    morph_mesh = spines_with_meshes.compound_spine_meshes_for_morphology()
    expected_mesh = trimesh.util.concatenate(spines_meshes)

    assert isinstance(morph_mesh, trimesh.Trimesh)
    assert_allclose(morph_mesh.vertices, expected_mesh.vertices)
    assert_allclose(morph_mesh.faces, expected_mesh.faces)


def test_centered_spine_meshes_for_morphology(spines_with_meshes, spines_meshes):
    morph_meshes = list(spines_with_meshes.centered_spine_meshes_for_morphology())

    assert len(morph_meshes) == len(spines_meshes)

    for mesh, expected_mesh in zip(morph_meshes, spines_meshes, strict=True):
        assert isinstance(mesh, trimesh.Trimesh)
        print(f"mesh: \n{mesh.vertices}, \nexpected_mesh: \n{expected_mesh.vertices}")
        assert_allclose(mesh.vertices, expected_mesh.vertices)
        assert_allclose(mesh.faces, expected_mesh.faces)


def test_centered_spine_meshes_for_morphology_centered(
    centered_spines_with_meshes, centered_spines_meshes
):
    morph_meshes = list(centered_spines_with_meshes.centered_spine_meshes_for_morphology())

    assert len(morph_meshes) == len(centered_spines_meshes)

    for mesh, expected_mesh in zip(morph_meshes, centered_spines_meshes, strict=True):
        assert isinstance(mesh, trimesh.Trimesh)
        print(f"mesh: \n{mesh.vertices}, \nexpected_mesh: \n{expected_mesh.vertices}")
        assert_allclose(mesh.vertices, expected_mesh.vertices)
        assert_allclose(mesh.faces, expected_mesh.faces)
