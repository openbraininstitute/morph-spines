"""This script creates a morphology-with-spines file with fake sample data."""

from pathlib import Path

import h5py
import numpy as np

morphology_filename = "morph_with_spines_schema.h5"
morphology_name = "01234"
output_dir = Path(f"{Path(__file__).parent.parent}/tests/data")
output_dir.mkdir(exist_ok=True)
output_file = Path(f"{output_dir}/{morphology_filename}")

random_spine_data = False
spine_compound_type = False

# Group /edges
dtypes = np.dtype(
    [
        ("afferent_surface_x", np.float64),
        ("afferent_surface_y", np.float64),
        ("afferent_surface_z", np.float64),
        ("afferent_center_x", np.float64),
        ("afferent_center_y", np.float64),
        ("afferent_center_z", np.float64),
        ("spine_id", np.int64),
        ("spine_morphology", "S16"),  # fixed-length string
        ("spine_length", np.float64),
        ("spine_orientation_vector_x", np.float64),
        ("spine_orientation_vector_y", np.float64),
        ("spine_orientation_vector_z", np.float64),
        ("spine_rotation_x", np.float64),
        ("spine_rotation_y", np.float64),
        ("spine_rotation_z", np.float64),
        ("spine_rotation_w", np.float64),
        ("afferent_section_id", np.int64),
        ("afferent_segment_id", np.int64),
        ("afferent_segment_offset", np.float64),
        ("afferent_section_pos", np.float64),
    ]
)

# Number of spines
num_spines = 2

# Create an empty structured array, to be filled with random or pre-defined data
data = np.empty(num_spines, dtype=dtypes)

# Fill the fields
if random_spine_data:
    data["afferent_surface_x"] = np.random.random(num_spines)
    data["afferent_surface_y"] = np.random.random(num_spines)
    data["afferent_surface_z"] = np.random.random(num_spines)

    data["afferent_center_x"] = np.random.random(num_spines)
    data["afferent_center_y"] = np.random.random(num_spines)
    data["afferent_center_z"] = np.random.random(num_spines)

    data["spine_id"] = np.array(range(num_spines), dtype=np.int64)
    data["spine_morphology"] = [morphology_name] * num_spines
    data["spine_length"] = np.random.random(num_spines)

    data["spine_orientation_vector_x"] = np.random.random(num_spines)
    data["spine_orientation_vector_y"] = np.random.random(num_spines)
    data["spine_orientation_vector_z"] = np.random.random(num_spines)

    data["spine_rotation_x"] = np.random.random(num_spines)
    data["spine_rotation_y"] = np.random.random(num_spines)
    data["spine_rotation_z"] = np.random.random(num_spines)
    data["spine_rotation_w"] = np.random.random(num_spines)

    data["afferent_section_id"] = np.random.randint(num_spines)
    data["afferent_segment_id"] = np.random.randint(num_spines)
    data["afferent_segment_offset"] = np.random.random(num_spines)
    data["afferent_section_pos"] = np.random.random(num_spines)

else:
    spines = np.arange(1, num_spines + 1, dtype=np.int64)

    data["afferent_surface_x"] = np.array(0.1 * spines, dtype=np.float64)
    data["afferent_surface_y"] = np.array(0.1 * spines + spines / 100, dtype=np.float64)
    data["afferent_surface_z"] = np.array(0.1 * spines + spines / 1000, dtype=np.float64)

    data["afferent_center_x"] = np.array(1.0 * spines + spines, dtype=np.float64)
    data["afferent_center_y"] = np.array(1.0 * spines + spines * 10, dtype=np.float64)
    data["afferent_center_z"] = np.array(1.0 * spines + spines * 100, dtype=np.float64)

    data["spine_id"] = np.arange(num_spines, dtype=np.int64)
    # data["spine_morphology"] =
    # np.array([morphology_name.encode("utf-8")] * num_spines, dtype="S32")
    data["spine_morphology"] = np.array([morphology_name] * num_spines, dtype="S32")
    data["spine_length"] = spines.astype(np.float64)

    data["spine_orientation_vector_x"] = np.array(0.1234 * spines, dtype=np.float64)
    data["spine_orientation_vector_y"] = np.array(0.2345 * spines, dtype=np.float64)
    data["spine_orientation_vector_z"] = np.array(0.3456 * spines, dtype=np.float64)

    data["spine_rotation_x"] = np.array(0.4567 * spines, dtype=np.float64)
    data["spine_rotation_y"] = np.array(0.5678 * spines, dtype=np.float64)
    data["spine_rotation_z"] = np.array(0.6789 * spines, dtype=np.float64)
    data["spine_rotation_w"] = np.array(0.7891 * spines, dtype=np.float64)

    data["afferent_section_id"] = np.array(10 + spines, dtype=np.int64)
    data["afferent_segment_id"] = np.array(100 + spines, dtype=np.int64)
    data["afferent_segment_offset"] = np.array(0.8901 * spines, dtype=np.float64)
    data["afferent_section_pos"] = np.array(0.9012 * spines, dtype=np.float64)

with h5py.File(output_file, "w") as h5_file:
    # Group /edges
    edges_grp = h5_file.create_group("edges")
    if spine_compound_type:
        # Create a single dataset table, compound type
        spine_table_key = str(f"/edges/{morphology_name}")
        edges_grp.create_dataset(spine_table_key, data=data)
    else:
        # Create as many datasets as columns in the table
        spine_table_grp_name = str(f"/edges/{morphology_name}")
        spine_table_grp = edges_grp.create_group(spine_table_grp_name)
        if data.dtype.names is not None:
            # We know it's not None, just making mypy happy
            for col_name in data.dtype.names:
                dset_name = str(f"{spine_table_grp_name}/{col_name}")
                spine_table_grp.create_dataset(dset_name, data=data[col_name])

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
