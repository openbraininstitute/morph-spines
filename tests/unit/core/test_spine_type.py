"""Tests for the SpineType enum."""

from morph_spines.core.spine_type import SpineType


def test_spine_type_values():
    """All expected spine types exist with correct string values."""
    assert SpineType.UNDEFINED.value == "undefined"
    assert SpineType.THIN.value == "thin"
    assert SpineType.LONG_THIN.value == "long_thin"
    assert SpineType.MUSHROOM.value == "mushroom"
    assert SpineType.STUBBY.value == "stubby"
    assert SpineType.FILOPODIUM.value == "filopodium"
    assert SpineType.BRANCHED.value == "branched"


def test_spine_type_is_string():
    """SpineType values are strings (str enum)."""
    assert isinstance(SpineType.MUSHROOM, str)
    assert SpineType.MUSHROOM == "mushroom"


def test_spine_type_from_string():
    """SpineType can be constructed from a string value."""
    assert SpineType("thin") == SpineType.THIN
    assert SpineType("branched") == SpineType.BRANCHED


def test_spine_type_count():
    """There are exactly 7 spine types."""
    assert len(SpineType) == 7
