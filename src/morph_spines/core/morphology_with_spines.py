"""Represents a neron morphology with spines.

Provides utility and data access to a representation of a
neuron morphology with individual spines.
"""

from dataclasses import dataclass

from neurom.core.morphology import Morphology

from morph_spines.core.soma import Soma
from morph_spines.core.spines import Spines


@dataclass
class MorphologyWithSpines:
    """Represents spiny neuron morphology.

    A container data class to access the information contained in the
    MorphologyWithSpines format.
    """

    morphology: Morphology
    soma: Soma
    spines: Spines
