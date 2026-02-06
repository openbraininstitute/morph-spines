from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent.parent / "data" / "morphology_with_spines"


@pytest.fixture(scope="session")
def single_morph_spines_file(test_data_dir):
    return test_data_dir / "single_neuron_spines.h5"


@pytest.fixture(scope="session")
def single_morph_spines_centered_file(test_data_dir):
    return test_data_dir / "single_neuron_spines_centered.h5"


@pytest.fixture(scope="session")
def single_morph_id():
    return "neuron_0"


@pytest.fixture(scope="session")
def multiple_morph_spines_file(test_data_dir):
    return test_data_dir / "multiple_neurons_spines.h5"


@pytest.fixture(scope="session")
def multiple_morph_spines_centered_file(test_data_dir):
    return test_data_dir / "multiple_neurons_spines_centered.h5"


@pytest.fixture(scope="session")
def multiple_morph_ids():
    return ["neuron_0", "neuron_1", "neuron_2"]


@pytest.fixture(scope="session")
def multiple_collections_centered_file(test_data_dir):
    return test_data_dir / "multiple_collections_centered.h5"


@pytest.fixture(scope="session")
def multiple_collections_moprh_ids():
    return ["neuron_0", "neuron_1"]


@pytest.fixture(scope="session")
def multiple_collections_coll_ids():
    return ["collection_0", "collection_1", "collection_2"]
