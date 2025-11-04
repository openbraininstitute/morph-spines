"""This script creates a morphology-with-spines file with fake sample data."""

import h5py
import numpy as np
from pathlib import Path


morphology_filename = "morph_with_spines_schema.h5"
morphology_name = "01234"
output_dir = Path(f"{Path(__file__).parent.parent}/tests/data")
output_dir.mkdir(exist_ok=True)
output_file = Path(f"{output_dir}/{morphology_filename}")

with h5py.File(output_file, "w") as h5_file:
    # Group /edges
    edges_grp = h5_file.create_group("edges")
    edges_id = edges_grp.create_group(morphology_name)
    edges_id.create_dataset("axis0", data=np.arange(3))
    edges_id.create_dataset("axis1", data=np.arange(3))
    edges_id.create_dataset("block0_items", data=np.arange(2))
    edges_id.create_dataset("block0_values", data=np.random.random((3, 2)))
    edges_id.create_dataset("block1_items", data=np.arange(2))
    edges_id.create_dataset("block1_values", data=np.random.random((3, 2)))
    edges_id.create_dataset("block2_items", data=np.array(2))
    edges_id.create_dataset("block2_values", data=np.random.random((2, 2)))

    # Group /morphology
    morph_grp = h5_file.create_group("morphology")
    morph_id = morph_grp.create_group(morphology_name)

    morph_points = np.array([
        [0, 0, 0, 1],
        [1, 1, 1, 1],
        [2, 2, 2, 4],
        [3, 3, 3, 4],
        [3, 3, 3, 4],
        [4, 4, 4, 4],
        [3, 3, 3, 5],
        [5, 5, 5, 5]
    ], dtype=np.float32)

    morph_structure = np.array([ [0, 1, -1], [2, 2, 0], [4, 2, 1], [6, 2, 1] ], dtype=np.int32)

    morph_id.create_dataset("points", data=morph_points)
    morph_id.create_dataset("structure", data=morph_structure)

    # Group /soma/meshes
    soma_grp = h5_file.create_group("soma")
    soma_meshes = soma_grp.create_group("meshes")
    soma_id = soma_meshes.create_group(morphology_name)

    # Create a triangular bipyramid (2 inverted pyramids, sharing its base)
    triangles = np.array([ [0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4] ])
    vertices = np.array([ [0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1] ])
    soma_id.create_dataset("triangles", data=triangles)
    soma_id.create_dataset("vertices", data=vertices)

    # Group /spines
    spines_grp = h5_file.create_group("spines")

    # Group /spines/meshes
    spines_meshes = spines_grp.create_group("meshes")
    spines_id = spines_meshes.create_group(morphology_name)
    spines_id.create_dataset("offsets", data=np.random.randint(0, 100, (2, 2)))
    spines_id.create_dataset("triangles", data=np.random.randint(0, 100, (2, 3)))
    spines_id.create_dataset("vertices", data=np.random.random((2, 3)))

    # Group /spines/skeletons
    spines_skel = spines_grp.create_group("skeletons")
    spines_skel_id = spines_skel.create_group(morphology_name)
    spines_skel_id.create_dataset("points", data=np.random.random((3, 4)))
    spines_skel_id.create_dataset("structure", data=np.random.randint(0, 10, (2, 3)))

print(f"{output_file} successfully created.")
