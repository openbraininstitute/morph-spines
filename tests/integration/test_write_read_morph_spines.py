import numpy as np
import pandas as pd
import pytest

from morph_spines.core.h5_schema import GRP_EDGES
from morph_spines.utils.morph_spine_loader import load_spine_table
from morph_spines.utils.morph_spine_writer import write_spine_table


@pytest.fixture
def valid_spine_table():
    """Create a minimal valid spine table with all mandatory columns."""
    n = 3
    return pd.DataFrame(
        {
            "afferent_surface_x": np.random.rand(n),
            "afferent_surface_y": np.random.rand(n),
            "afferent_surface_z": np.random.rand(n),
            "afferent_center_x": np.random.rand(n),
            "afferent_center_y": np.random.rand(n),
            "afferent_center_z": np.random.rand(n),
            "spine_morphology": ["morph_a"] * n,
            "spine_id": np.array([0, 1, 2], dtype=np.uint32),
            "spine_length": np.random.rand(n),
            "spine_orientation_vector_x": np.random.rand(n),
            "spine_orientation_vector_y": np.random.rand(n),
            "spine_orientation_vector_z": np.random.rand(n),
            "spine_rotation_x": np.random.rand(n),
            "spine_rotation_y": np.random.rand(n),
            "spine_rotation_z": np.random.rand(n),
            "spine_rotation_w": np.random.rand(n),
            "afferent_section_id": np.array([1, 2, 3], dtype=np.uint32),
            "afferent_segment_id": np.array([0, 1, 2], dtype=np.int32),
            "afferent_segment_offset": np.random.rand(n),
            "afferent_section_pos": np.random.rand(n),
        }
    )


class TestWriteReadRoundtrip:
    def test_spine_table_readable_by_loader(self, tmp_path, valid_spine_table):
        f = tmp_path / "output.h5"
        write_spine_table(str(f), "neuron_01", valid_spine_table)

        loaded = load_spine_table(str(f), f"{GRP_EDGES}/neuron_01")

        assert set(loaded.columns) == set(valid_spine_table.columns)
        assert len(loaded) == len(valid_spine_table)
        for col in valid_spine_table.columns:
            loaded_arr = loaded[col].to_numpy()
            expected_arr = valid_spine_table[col].to_numpy()
            # String columns: compare as strings
            if expected_arr.dtype == object or expected_arr.dtype.kind in ("U", "S", "O"):
                np.testing.assert_array_equal(
                    loaded_arr.astype(str),
                    expected_arr.astype(str),
                )
            elif loaded_arr.dtype.kind in ("U", "S", "O"):
                np.testing.assert_array_equal(
                    loaded_arr.astype(str),
                    expected_arr.astype(str),
                )
            else:
                np.testing.assert_array_almost_equal(
                    loaded_arr.astype(float),
                    expected_arr.astype(float),
                )
