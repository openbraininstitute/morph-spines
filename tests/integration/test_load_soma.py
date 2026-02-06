from morph_spines import Soma
from morph_spines.utils.morph_spine_loader import load_soma


def test_load_soma(single_morph_spines_file):
    soma = load_soma(single_morph_spines_file)

    assert isinstance(soma, Soma)
