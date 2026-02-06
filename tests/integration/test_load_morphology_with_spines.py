import trimesh

from morph_spines.core.morphology_with_spines import MorphologyWithSpines
from morph_spines.utils.morph_spine_loader import load_morphology_with_spines


def test_load_morphology_with_spines_single_neuron(single_morph_spines_file):
    morph_with_spines = load_morphology_with_spines(
        single_morph_spines_file, spines_are_centered=False
    )
    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert morph_with_spines.spines.spine_count == 4


def test_load_morphology_with_spines_single_neuron_centered(single_morph_spines_centered_file):
    morph_with_spines = load_morphology_with_spines(
        single_morph_spines_centered_file, spines_are_centered=True
    )
    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert morph_with_spines.spines.spine_count == 4


def test_load_morphology_with_spines_multiple_neurons(
    multiple_morph_spines_file, multiple_morph_ids
):
    morph_with_spines = load_morphology_with_spines(
        multiple_morph_spines_file, morphology_name=multiple_morph_ids[0], spines_are_centered=False
    )
    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert morph_with_spines.spines.spine_count == 4


def test_load_morphology_with_spines_multiple_neurons_centered(
    multiple_morph_spines_centered_file, multiple_morph_ids
):
    morph_with_spines = load_morphology_with_spines(
        multiple_morph_spines_centered_file,
        morphology_name=multiple_morph_ids[0],
        spines_are_centered=True,
    )
    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert morph_with_spines.spines.spine_count == 4


def test_load_morphology_with_spines_from_collection_centered(
    multiple_collections_centered_file, multiple_collections_moprh_ids
):
    morph_with_spines = load_morphology_with_spines(
        multiple_collections_centered_file,
        morphology_name=multiple_collections_moprh_ids[0],
        spines_are_centered=True,
    )
    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert morph_with_spines.spines.spine_count == 6


def test_load_morphology_with_spines_load_meshes(multiple_morph_spines_file, multiple_morph_ids):
    morph_with_spines = load_morphology_with_spines(
        multiple_morph_spines_file,
        morphology_name=multiple_morph_ids[0],
        spines_are_centered=False,
        load_meshes=True,
    )

    spine_meshes = list(morph_with_spines.spines.spine_meshes_for_morphology())

    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert len(spine_meshes) == 4
    assert isinstance(spine_meshes[0], trimesh.Trimesh)
    assert isinstance(
        morph_with_spines.spines.compound_spine_meshes_for_morphology(), trimesh.Trimesh
    )


def test_load_morphology_with_spines_load_meshes_centered(
    multiple_morph_spines_centered_file, multiple_morph_ids
):
    morph_with_spines = load_morphology_with_spines(
        multiple_morph_spines_centered_file,
        morphology_name=multiple_morph_ids[0],
        spines_are_centered=True,
        load_meshes=True,
    )

    spine_meshes = list(morph_with_spines.spines.spine_meshes_for_morphology())

    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert len(spine_meshes) == 4
    assert isinstance(spine_meshes[0], trimesh.Trimesh)
    assert isinstance(
        morph_with_spines.spines.compound_spine_meshes_for_morphology(), trimesh.Trimesh
    )


def test_load_morphology_with_spines_from_collection_load_meshes_centered(
    multiple_collections_centered_file, multiple_collections_moprh_ids
):
    morph_with_spines = load_morphology_with_spines(
        multiple_collections_centered_file,
        morphology_name=multiple_collections_moprh_ids[0],
        spines_are_centered=True,
        load_meshes=True,
    )

    spine_meshes = list(morph_with_spines.spines.spine_meshes_for_morphology())

    assert isinstance(morph_with_spines, MorphologyWithSpines)
    assert len(spine_meshes) == 6
    assert isinstance(spine_meshes[0], trimesh.Trimesh)
    assert isinstance(
        morph_with_spines.spines.compound_spine_meshes_for_morphology(), trimesh.Trimesh
    )
