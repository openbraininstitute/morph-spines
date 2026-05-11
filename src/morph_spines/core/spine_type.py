"""Defines the spine type classification.

Spine types are based on morphological characteristics such as length, width,
and the presence of a distinct head and neck.
"""

from enum import StrEnum


class SpineType(StrEnum):
    """Classification of dendritic spine morphological types.

    Each spine type is characterized by distinct geometric features:
    - THIN: narrow neck with small head
    - LONG_THIN: long, narrow neck with small head
    - MUSHROOM: large bulbous head on a narrow neck
    - STUBBY: width larger than length, no clear distinction between head and neck
    - FILOPODIUM: long thin protrusion without a distinct head
    - BRANCHED: 2 or more heads sharing a common neck
    - UNDEFINED: Type not determined or not applicable
    """

    UNDEFINED = "undefined"
    THIN = "thin"
    LONG_THIN = "long_thin"
    MUSHROOM = "mushroom"
    STUBBY = "stubby"
    FILOPODIUM = "filopodium"
    BRANCHED = "branched"
