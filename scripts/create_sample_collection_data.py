"""This script creates a morphology-with-spines file with fake sample data.

See: <python create_sample_data.py -h> for help on usage.
"""

import argparse
import os

import h5py
import numpy as np
from numpy.typing import NDArray

# Global variables
# Format versions
spine_table_version = np.array([1, 0], dtype=np.uint32)
neuron_morphology_version = np.array([1, 3], dtype=np.uint32)
neuron_morphology_family = np.array([0], dtype=np.uint32)
spine_morphology_version = np.array([1, 3], dtype=np.uint32)  # FIXME: 1.4 once morphio supports it
spine_morphology_family = np.array([0], dtype=np.uint32)  # FIXME: 3 once morphio supports it

# Spine table columns
dtypes = np.dtype(
    [
        ("afferent_surface_x", np.float64),
        ("afferent_surface_y", np.float64),
        ("afferent_surface_z", np.float64),
        ("afferent_center_x", np.float64),
        ("afferent_center_y", np.float64),
        ("afferent_center_z", np.float64),
        ("spine_id", np.int64),
        ("spine_morphology", "S32"),  # fixed-length string
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


# Spines meshes library
meshes_library = {
    ("tetrahedron", "points"): [
        [0.0, 0.0, 0.0],
        [-0.5, 1.0, -0.5],
        [0.5, 1.0, -0.5],
        [0.0, 1.0, 0.5],
    ],
    ("tetrahedron", "shift"): [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
    ("tetrahedron", "triangles"): [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 3, 2]],
    ("pyramid", "points"): [
        [0.0, 0.0, 0.0],
        [-0.5, 1.0, -0.5],
        [0.5, 1.0, -0.5],
        [0.5, 1.0, 0.5],
        [-0.5, 1.0, 0.5],
    ],
    ("pyramid", "shift"): [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    ("pyramid", "triangles"): [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 3, 2], [1, 4, 3]],
    ("prism", "points"): [
        [-0.5, 0.0, -0.5],
        [0.5, 0.0, -0.5],
        [0.0, 0.0, 0.5],
        [-0.5, 1.0, -0.5],
        [0.5, 1.0, -0.5],
        [0.0, 1.0, 0.5],
    ],
    ("prism", "shift"): [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    ("prism", "triangles"): [
        [0, 1, 2],
        [3, 5, 4],
        [0, 4, 1],
        [0, 3, 4],
        [1, 5, 2],
        [1, 4, 5],
        [2, 3, 0],
        [2, 5, 3],
    ],
}

mesh_library_shapes_and_offsets = [("tetrahedron", 4, 4), ("pyramid", 5, 6), ("prism", 6, 8)]


def generate_spines_data(
    neuron_idx: int, coll_names: list[str], num_morph_spines_per_coll: int
) -> NDArray:
    """Create a 2D array with pre-defined values for spine data.

    Args:
        neuron_idx: Morphology index for which the data will be generated
        coll_names: Names of spine collections
        num_morph_spines_per_coll: number of spines per collection to be added to the neuron

    Returns: 2-dimensional array with pre-defined values for spine data
    """
    # Create an empty structured array to be filled with pre-defined values
    num_spines = len(coll_names) * num_morph_spines_per_coll
    data = np.empty(num_spines, dtype=dtypes)

    spines = np.arange(1, num_spines + 1, dtype=np.int64)

    afferent_surface = [
        [2.0 + neuron_idx, 5.0 + neuron_idx] * num_morph_spines_per_coll,  # X
        [2.0, 5.0] * num_morph_spines_per_coll,  # Y
        [2.0, 5.0] * num_morph_spines_per_coll,  # Z
    ]

    afferent_surface_x = []
    afferent_surface_y = []
    afferent_surface_z = []
    for c in range(len(coll_names)):
        for m in range(num_morph_spines_per_coll):
            afferent_surface_x.append(afferent_surface[0][m] * (m + 1))
            afferent_surface_y.append(afferent_surface[1][m] * (c + 1))
            afferent_surface_z.append(afferent_surface[2][m] * (c + m + 1))

    data["afferent_surface_x"] = np.array(afferent_surface_x, dtype=np.float64)
    data["afferent_surface_y"] = np.array(afferent_surface_y, dtype=np.float64)
    data["afferent_surface_z"] = np.array(afferent_surface_y, dtype=np.float64)

    data["afferent_center_x"] = data["afferent_surface_x"]
    data["afferent_center_y"] = data["afferent_surface_y"]
    data["afferent_center_z"] = data["afferent_surface_z"]

    data["spine_id"] = np.repeat(
        np.arange(num_morph_spines_per_coll, dtype=np.int64), len(coll_names)
    )
    data["spine_morphology"] = np.tile(
        np.array([coll_names], dtype="S32"), num_morph_spines_per_coll
    )
    data["spine_length"] = np.repeat(np.arange(1, num_morph_spines_per_coll + 1), len(coll_names))

    data["spine_orientation_vector_x"] = np.repeat(np.array([0.0], dtype=np.float64), num_spines)
    data["spine_orientation_vector_y"] = np.repeat(np.array([1.0], dtype=np.float64), num_spines)
    data["spine_orientation_vector_z"] = np.repeat(np.array([0.0], dtype=np.float64), num_spines)

    data["spine_rotation_x"] = np.repeat(np.array([0.0], dtype=np.float64), num_spines)
    data["spine_rotation_y"] = np.repeat(np.array([0.0], dtype=np.float64), num_spines)
    data["spine_rotation_z"] = np.repeat(np.array([0.0], dtype=np.float64), num_spines)
    data["spine_rotation_w"] = np.repeat(np.array([1.0], dtype=np.float64), num_spines)

    data["afferent_section_id"] = np.array([2] * spines, dtype=np.int64)
    data["afferent_segment_id"] = np.array([0] * spines, dtype=np.int64)
    data["afferent_segment_offset"] = np.array([0.5] * spines, dtype=np.float64)
    data["afferent_section_pos"] = np.array([2] * spines, dtype=np.float64)

    return data


def generate_neuron_skeleton(neuron_idx: int) -> dict[str, np.ndarray]:
    """Generate a neuron skeleton whose coordinates are based on its index.

    In order to make it simple, the same morphology is always created. The only difference is that
    the X axis is shifted by +neuron_idx units.

    Args:
        neuron_idx: The neuron's index

    Returns: A dictionary with the neuron skeleton (points + structure arrays)
    """
    morph_points = np.array(
        [
            [0, 0, 0, 1],
            [1, 1, 1, 1],
            [2, 2, 2, 0.5],
            [3, 2, 3, 0.5],
            [3, 2, 3, 0.5],
            [4, 3, 3, 0.3],
            [3, 2, 3, 0.5],
            [5, 5, 5, 0.3],
        ],
        dtype=np.float32,
    )

    morph_structure = np.array([[0, 1, -1], [2, 2, 0], [4, 2, 1], [6, 2, 1]], dtype=np.int32)

    coord_shift = np.array([neuron_idx, 0, 0, 0], dtype=np.float32)
    morph_points = (morph_points + coord_shift).astype(np.float32)

    neuron_skeleton = {"points": morph_points, "structure": morph_structure}

    return neuron_skeleton


def generate_soma_mesh(neuron_idx: int) -> dict[str, np.ndarray]:
    """Generate a mesh of a neuron soma whose coordinates are based on its index.

    In order to make it simple, the same mesh is always created. The only difference is that
    the X axis is shifted by +neuron_idx units.

    Args:
        neuron_idx: The neuron's index

    Returns: A dictionary with the neuron's soma mesh (points + vertices arrays)
    """
    # Create a triangular bipyramid (2 inverted pyramids, sharing its base)
    triangles = np.array(
        [[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [5, 2, 1], [5, 3, 2], [5, 4, 3], [5, 1, 4]]
    )
    vertices = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])

    coord_shift = np.array([neuron_idx, 0, 0], dtype=np.float32)
    vertices = vertices + coord_shift

    soma_mesh = {"triangles": triangles, "vertices": vertices}

    return soma_mesh


def generate_spines_skeletons(coll_idx: int, num_spines: int) -> dict[str, np.ndarray]:
    """Generate multiple spine skeletons whose coordinates are based on its spine index.

    In order to make it simple, the same spine skeletons are always created. The only difference is
    that the Y axis is shifted by +spine_idx units (longer spines).
    Note: num_spines can only be 1 or 2
    Args:
        coll_idx: The collection's index
        num_spines: The number of spines of the collection (valid values are 1 or 2 only)

    Returns: A dictionary with the collection's spines skeletons (points + structure arrays)
    """
    if num_spines != 1 and num_spines != 2:
        raise ValueError("Unsupported num_spines value: can only be 1 or 2")

    points = np.array(
        [
            [0, 0, 0, 0.1],  # spine 0 start (x, y, z, d)
            [0, 1, 0, 2.0],  # spine 0 end (x, y, z, d)
            [0, 0, 0, 0.1],  # spine 1 start (x, y, z, d)
            [0, 2, 0, 2.0],  # spine 1 end (x, y, z, d)
        ],
        dtype=np.float32,
    )

    structure = np.array([[0, 2, -1], [2, 2, -1]])

    points = points[0:2] if num_spines == 1 else points
    structure = structure[0:1] if num_spines == 1 else structure

    # coord_shift = np.array([coll_idx, 0, 0, 0], dtype=np.float32)
    # points = points + coord_shift

    spines_skeletons = {"points": points, "structure": structure}

    return spines_skeletons


def generate_spines_meshes(coll_idx: int, num_spines: int) -> dict[str, np.ndarray]:
    """Generate multiple spine meshes whose shapes are based on its collection index.

    In order to make it simple, similar spine meshes are always created. The only difference is
    that their shape alternates between: tetrahedron, pyramid and prism.
    Note: num_spines can only be 1 or 2
    Args:
        coll_idx: The collection's index
        num_spines: The number of spines of the collection (valid values are 1 or 2 only)

    Returns: A dictionary with the collection's spines meshes (offsets + points + vertices arrays)
    """
    if num_spines != 1 and num_spines != 2:
        raise ValueError("Unsupported num_spines value: can only be 1 or 2")

    vertices = []
    # Offset format: [vertices_offset, triangles_offset]
    offsets = [[0, 0]]
    # Shape = (geometry, vertices_offset, triangles_offset)
    shape = mesh_library_shapes_and_offsets[coll_idx % len(mesh_library_shapes_and_offsets)]

    # For each spine, make it one unit larger than the previous one in the Y axis
    for i in range(num_spines):
        shift = np.array(meshes_library[(shape[0], "shift")], dtype=np.float64) * i
        points = np.array(meshes_library[(shape[0], "points")], dtype=np.float64) + shift
        vertices.append(points)
        offsets.append([shape[1] * (i + 1), shape[2] * (i + 1)])

    spines_meshes = {
        "offsets": np.vstack(offsets),
        "triangles": np.tile(
            np.array(meshes_library[(shape[0], "triangles")], dtype=np.int32), (num_spines, 1)
        ),
        "vertices": np.vstack(vertices),
    }

    return spines_meshes


def write_neuron_data(output_file: str, neuron_name: str, data: dict) -> None:
    """Write the collection of neuron and spine data to file.

    Args:
        output_file: Filepath to output file that will be created
        neuron_name: Morphology ID under which the data will be written
        data: dictionary with H5 groups/datasets to be created in the file

    Returns: None
    """
    spine_table = data[(neuron_name, "spine_table")]
    neuron_skeleton = data[(neuron_name, "neuron_skeleton")]
    soma_mesh = data[(neuron_name, "soma_mesh")]

    mode = "a" if os.path.exists(output_file) else "w"

    with h5py.File(output_file, mode) as h5_file:
        # Group /edges
        # Get group if it exists or create it otherwise
        edges_grp = h5_file.require_group("edges")

        # Neuron subgroup must not exist, as neuron IDs should be unique
        spine_table_grp_name = str(f"/edges/{neuron_name}")
        spine_table_grp = edges_grp.create_group(spine_table_grp_name)

        # Spine table metadata
        edges_metadata = spine_table_grp.create_group("metadata")
        edges_metadata.attrs["version"] = spine_table_version

        # Create as many datasets as columns in the table
        if spine_table.dtype.names is not None:
            # We know it's not None, just making mypy happy
            for col_name in spine_table.dtype.names:
                dset_name = str(f"{spine_table_grp_name}/{col_name}")
                spine_table_grp.create_dataset(dset_name, data=spine_table[col_name])

        # Group /morphology
        # Create top group if it doesn't exist; create neuron ID group inside
        morph_grp = h5_file.require_group("morphology")
        morph_id = morph_grp.create_group(neuron_name)

        # Morphology metadata
        morph_metadata = morph_id.create_group("metadata")
        morph_metadata.attrs["cell_family"] = neuron_morphology_family
        morph_metadata.attrs["version"] = neuron_morphology_version

        morph_id.create_dataset("points", data=neuron_skeleton["points"])
        morph_id.create_dataset("structure", data=neuron_skeleton["structure"])

        # Group /soma/meshes
        # Create top groups if they don't exist; create neuron ID group inside
        soma_grp = h5_file.require_group("soma")
        soma_meshes_grp = soma_grp.require_group("meshes")
        soma_id = soma_meshes_grp.create_group(neuron_name)

        soma_id.create_dataset("triangles", data=soma_mesh["triangles"])
        soma_id.create_dataset("vertices", data=soma_mesh["vertices"])

    print(f'Successfully added morphology-with-spines "{neuron_name}".')


def write_spines_data(output_file: str, coll_names: list[str], data: dict) -> None:
    """Write the collection of neuron and spine data to file.

    Args:
        output_file: Filepath to output file that will be created
        coll_names: Morphology ID under which the data will be written
        data: dictionary with H5 groups/datasets to be created in the file

    Returns: None
    """
    if len(coll_names) == 0:
        print("No spine collections were found, nothing was written to the file")
        return

    mode = "a" if os.path.exists(output_file) else "w"
    with h5py.File(output_file, mode) as h5_file:
        for coll_name in coll_names:
            spines_skeletons = data[(coll_name, "spines_skeletons")]
            spines_meshes = data[(coll_name, "spines_meshes")]

            # Group /spines
            # Create top groups if they don't exist; create collections groups inside
            spines_grp = h5_file.require_group("spines")

            # Group /spines/meshes
            spines_meshes_grp = spines_grp.require_group("meshes")
            coll_meshes_group = spines_meshes_grp.create_group(coll_name)

            coll_meshes_group.create_dataset("offsets", data=spines_meshes["offsets"])
            coll_meshes_group.create_dataset("triangles", data=spines_meshes["triangles"])
            coll_meshes_group.create_dataset("vertices", data=spines_meshes["vertices"])

            # Group /spines/skeletons
            spines_skel = spines_grp.require_group("skeletons")
            coll_skel_group = spines_skel.create_group(coll_name)

            # Spine skeleton metadata
            spine_metadata = coll_skel_group.create_group("metadata")
            spine_metadata.attrs["cell_family"] = spine_morphology_family
            spine_metadata.attrs["version"] = spine_morphology_version

            coll_skel_group.create_dataset("points", data=spines_skeletons["points"])
            coll_skel_group.create_dataset("structure", data=spines_skeletons["structure"])

    print(f'Successfully added the following spines collections: "{coll_names}".')


def create_sample_data(output_file: str, num_neurons: int = 1, num_colls: int = 2):
    """Generate the sample data and write it to the given output file.

    The spines are grouped into a number of collections, defined by num_colls.
    Each neuron will have 2 spines of each collection.

    Args:
        output_file: Filepath to output file that will be created
        num_neurons: Number of neurons to be written in the file
        num_colls: Number of spine collections

    Returns: None
    """
    spines_data = {}
    num_morph_spines_per_coll = 2

    coll_names = [f"coll_{i}" for i in range(num_colls)]

    for i, coll in enumerate(coll_names):
        spines_skeletons = generate_spines_skeletons(i, num_morph_spines_per_coll)
        spines_meshes = generate_spines_meshes(i, num_morph_spines_per_coll)
        spines_data[(coll, "spines_skeletons")] = spines_skeletons
        spines_data[(coll, "spines_meshes")] = spines_meshes

    for i in range(num_neurons):
        neuron_name = f"neuron_{i}"
        spine_table = generate_spines_data(i, coll_names, num_morph_spines_per_coll)

        neuron_skeleton = generate_neuron_skeleton(i)
        soma_mesh = generate_soma_mesh(i)

        data = {
            (neuron_name, "spine_table"): spine_table,
            (neuron_name, "neuron_skeleton"): neuron_skeleton,
            (neuron_name, "soma_mesh"): soma_mesh,
        }

        write_neuron_data(output_file, neuron_name, data)

    write_spines_data(output_file, coll_names, spines_data)

    print(f"{output_file} successfully created.")


def main() -> None:
    """Main function.

    Returns: None
    """
    parser = argparse.ArgumentParser(
        description="Generate sample collection data in morphology-with-spines format"
    )

    # Output file (string)
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # Number of neurons (int)
    parser.add_argument("-nneurons", type=int, default=1, help="Number of neurons")

    # Number of spine collections (int)
    parser.add_argument("-ncolls", type=int, default=1, help="Number of spine collections")

    # Number of spines (int)
    # parser.add_argument("-nspines", type=int, default=2, help="Number of spines per neuron")

    args = parser.parse_args()

    print("Output file:", args.output)
    print("Number of neurons:", args.nneurons)
    print("Number of spine collections:", args.ncolls)

    create_sample_data(args.output, args.nneurons, args.ncolls)


if __name__ == "__main__":
    main()
