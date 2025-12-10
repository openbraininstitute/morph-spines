"""This script creates a morphology-with-spines file with fake sample data."""

from pathlib import Path

import h5py
import numpy as np
import pandas as pd

morphology_filename = "morph_with_spines_schema_v0.1.h5"
morphology_name = "01234"
output_dir = Path(f"{Path(__file__).parent.parent}/tests/data")
output_dir.mkdir(exist_ok=True)
output_file = Path(f"{output_dir}/{morphology_filename}")

# Group /edges
# edges_grp = h5_file.create_group("edges")
# edges_id = edges_grp.create_group(morphology_name)

# Pandas dataframe columns
columns = [
    "afferent_surface_x",
    "afferent_surface_y",
    "afferent_surface_z",
    "afferent_center_x",
    "afferent_center_y",
    "afferent_center_z",
    "spine_id",
    "spine_morphology",
    "spine_length",
    "spine_orientation_vector_x",
    "spine_orientation_vector_y",
    "spine_orientation_vector_z",
    "spine_rotation_x",
    "spine_rotation_y",
    "spine_rotation_z",
    "spine_rotation_w",
    "afferent_section_id",
    "afferent_segment_id",
    "afferent_segment_offset",
    "afferent_section_pos",
]

# Number of spines
num_spines = 2

spine_table_version = np.array([0, 1], dtype=np.uint32)
random_df_data = False

if random_df_data:
    df = pd.DataFrame(
        {
            "afferent_surface_x": np.random.random(num_spines),
            "afferent_surface_y": np.random.random(num_spines),
            "afferent_surface_z": np.random.random(num_spines),
            "afferent_center_x": np.random.random(num_spines),
            "afferent_center_y": np.random.random(num_spines),
            "afferent_center_z": np.random.random(num_spines),
            "spine_id": np.array(range(num_spines), dtype=np.int64),
            "spine_morphology": [morphology_name] * num_spines,
            "spine_length": np.random.random(num_spines),
            "spine_orientation_vector_x": np.random.random(num_spines),
            "spine_orientation_vector_y": np.random.random(num_spines),
            "spine_orientation_vector_z": np.random.random(num_spines),
            "spine_rotation_x": np.random.random(num_spines),
            "spine_rotation_y": np.random.random(num_spines),
            "spine_rotation_z": np.random.random(num_spines),
            "spine_rotation_w": np.random.random(num_spines),
            "afferent_section_id": np.random.randint(num_spines),
            "afferent_segment_id": np.random.randint(num_spines),
            "afferent_segment_offset": np.random.random(num_spines),
            "afferent_section_pos": np.random.random(num_spines),
        }
    )
else:
    spines = np.array(range(num_spines), dtype=np.int64) + 1
    df = pd.DataFrame(
        {
            "afferent_surface_x": np.array([0.1 * i for i in spines], dtype=np.float64),
            "afferent_surface_y": np.array([0.1 * i + i / 100 for i in spines], dtype=np.float64),
            "afferent_surface_z": np.array([0.1 * i + i / 1000 for i in spines], dtype=np.float64),
            "afferent_center_x": np.array([1.0 * i + i for i in spines], dtype=np.float64),
            "afferent_center_y": np.array([1.0 * i + i * 10 for i in spines], dtype=np.float64),
            "afferent_center_z": np.array([1.0 * i + i * 100 for i in spines], dtype=np.float64),
            "spine_id": np.array(range(num_spines), dtype=np.int64),
            "spine_morphology": [morphology_name] * num_spines,
            "spine_length": np.array([i for i in spines], dtype=np.float64),
            "spine_orientation_vector_x": np.array([0.1234 * i for i in spines], dtype=np.float64),
            "spine_orientation_vector_y": np.array([0.2345 * i for i in spines], dtype=np.float64),
            "spine_orientation_vector_z": np.array([0.3456 * i for i in spines], dtype=np.float64),
            "spine_rotation_x": np.array([0.4567 * i for i in spines], dtype=np.float64),
            "spine_rotation_y": np.array([0.5678 * i for i in spines], dtype=np.float64),
            "spine_rotation_z": np.array([0.6789 * i for i in spines], dtype=np.float64),
            "spine_rotation_w": np.array([0.7891 * i for i in spines], dtype=np.float64),
            "afferent_section_id": np.array([10 + i for i in spines], dtype=np.int64),
            "afferent_segment_id": np.array([100 + i for i in spines], dtype=np.int64),
            "afferent_segment_offset": np.array([0.8901 * i for i in spines], dtype=np.float64),
            "afferent_section_pos": np.array([0.9012 * i for i in spines], dtype=np.float64),
        }
    )

key = str(f"/edges/{morphology_name}")

df.to_hdf(output_file, key=key, mode="w")

with h5py.File(output_file, "a") as h5_file:
    # Spine table metadata (edges)
    edges_grp = h5_file[f"edges/{morphology_name}"]
    metadata = edges_grp.create_group("metadata")
    metadata.attrs["version"] = spine_table_version

    # Group /morphology
    morph_grp = h5_file.create_group("morphology")
    morph_id = morph_grp.create_group(morphology_name)

    morph_points = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 4],
            [3, 2, 3, 4],
            [3, 2, 3, 4],
            [4, 3, 3, 4],
            [3, 2, 3, 5],
            [5, 5, 5, 5],
        ],
        dtype=np.float32,
    )

    morph_structure = np.array([[0, 1, -1], [2, 2, 0], [4, 2, 1], [6, 2, 1]], dtype=np.int32)

    morph_id.create_dataset("points", data=morph_points)
    morph_id.create_dataset("structure", data=morph_structure)

    # Group /soma/meshes
    soma_grp = h5_file.create_group("soma")
    soma_meshes = soma_grp.create_group("meshes")
    soma_id = soma_meshes.create_group(morphology_name)

    # Create a triangular bipyramid (2 inverted pyramids, sharing its base)
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]]
    )
    vertices = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])
    soma_id.create_dataset("triangles", data=triangles)
    soma_id.create_dataset("vertices", data=vertices)

    # Group /spines
    spines_grp = h5_file.create_group("spines")

    # Group /spines/meshes
    spines_meshes = spines_grp.create_group("meshes")
    spines_id = spines_meshes.create_group(morphology_name)

    # Offset format: [vertices_offset, triangles_offset]
    offsets = np.array(
        [
            [0, 0],
            [5, 6],
            [9, 10],
        ]
    )

    # Triangles-vertices pair examples:
    # square-based pyramid shaped spine, center at (0, 0, 0):
    #   triangles = [ [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3] ]
    #   vertices = [ [0, 0, 2], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0] ]
    # triangle-based pyramid shaped spine, center at (0, 0, 0)
    #   triangles = [ [0, 2, 1], [0, 3, 2], [0, 1, 3], [1, 2, 3] ]
    #   vertices = [ [0, 0, 4], [0, 2, 0], [2, -2, 0], [-2, -2, 0] ]
    # square-based pyramid shaped spine, apex at (2, 2, 2)
    #   triangles = [ [0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [1, 2, 3], [1, 3, 4] ]
    #   vertices = [ [2, 2, 2], [3, 2, 4], [2, 3, 4], [1, 2, 4], [2, 1, 4] ]
    # triangle-based pyramid shaped spine, apex at (5, 5, 5)
    #   triangles = [ [0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2] ]
    #   vertices = [ [5, 5, 5], [5, 7, 9], [7, 3, 9], [3, 3, 9] ]
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 4, 3],
            [0, 1, 4],
            [1, 2, 3],
            [1, 3, 4],  # end of square-based pyramid shaped spine, apex at (2, 2, 2)
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
            [1, 3, 2],  # end of triangle-based pyramid shaped spine, apex at (5, 5, 5)
        ]
    )
    vertices = np.array(
        [
            [2, 2, 2],
            [3, 2, 4],
            [2, 3, 4],
            [1, 2, 4],
            [2, 1, 4],  # end of square-based pyramid shaped spine, apex at (2, 2, 2)
            [5, 5, 5],
            [5, 7, 9],
            [7, 3, 9],
            [3, 3, 9],  # end of triangle-based pyramid shaped spine, apex at (5, 5, 5)
        ]
    )

    spines_id.create_dataset("offsets", data=offsets)
    spines_id.create_dataset("triangles", data=triangles)
    spines_id.create_dataset("vertices", data=vertices)

    # Group /spines/skeletons
    spines_skel = spines_grp.create_group("skeletons")
    spines_skel_id = spines_skel.create_group(morphology_name)

    points = np.array(
        [
            [2, 2, 2, 0.1],  # spine 0 start (x, y, z, d)
            [2, 2, 4, 2.0],  # spine 0 end (x, y, z, d)
            [5, 5, 5, 0.1],  # spine 1 start (x, y, z, d)
            [5, 5, 9, 2.0],  # spine 1 end (x, y, z, d)
        ]
    )

    structure = np.array([[2, 2, -1], [5, 2, -1]])

    spines_skel_id.create_dataset("points", data=points)
    spines_skel_id.create_dataset("structure", data=structure)

print(f"{output_file} successfully created.")
