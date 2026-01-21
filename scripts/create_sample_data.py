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


def generate_random_spine_data(neuron_name: str, num_spines: int) -> NDArray:
    """Create a 2D array with random spine data.

    Args:
        neuron_name: Morphology ID for which the data will be generated
        num_spines: number of spines

    Returns: 2-dimensional array with random spine data
    """
    # Create an empty structured array to be filled with random data
    data = np.empty(num_spines, dtype=dtypes)

    # Fill the fields
    data["afferent_surface_x"] = np.random.random(num_spines)
    data["afferent_surface_y"] = np.random.random(num_spines)
    data["afferent_surface_z"] = np.random.random(num_spines)

    data["afferent_center_x"] = np.random.random(num_spines)
    data["afferent_center_y"] = np.random.random(num_spines)
    data["afferent_center_z"] = np.random.random(num_spines)

    data["spine_id"] = np.array(range(num_spines), dtype=np.int64)
    data["spine_morphology"] = np.array([neuron_name] * num_spines, dtype="S32")
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

    return data


def generate_spine_data(neuron_name: str, num_spines: int) -> NDArray:
    """Create a 2D array with pre-defined values for spine data.

    Args:
        neuron_name: Morphology ID for which the data will be generated
        num_spines: number of spines

    Returns: 2-dimensional array with pre-defined values for spine data
    """
    # Create an empty structured array to be filled with pre-defined values
    data = np.empty(num_spines, dtype=dtypes)

    spines = np.arange(1, num_spines + 1, dtype=np.int64)

    data["afferent_surface_x"] = np.array(0.1 * spines, dtype=np.float64)
    data["afferent_surface_y"] = np.array(0.1 * spines + spines / 100, dtype=np.float64)
    data["afferent_surface_z"] = np.array(0.1 * spines + spines / 1000, dtype=np.float64)

    data["afferent_center_x"] = np.array(1.0 * spines + spines, dtype=np.float64)
    data["afferent_center_y"] = np.array(1.0 * spines + spines * 10, dtype=np.float64)
    data["afferent_center_z"] = np.array(1.0 * spines + spines * 100, dtype=np.float64)

    data["spine_id"] = np.arange(num_spines, dtype=np.int64)
    data["spine_morphology"] = np.array([neuron_name] * num_spines, dtype="S32")
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


def generate_spine_skeletons(neuron_idx: int, num_spines: int) -> dict[str, np.ndarray]:
    """Generate multiple spine skeletons whose coordinates are based on its neuron index.

    In order to make it simple, the same spine skeletons are always created. The only difference is
    that the X axis is shifted by +neuron_idx units.
    Note: num_spines can only be 1 or 2
    Args:
        neuron_idx: The neuron's index
        num_spines: The number of spines of the neuron (valid values are 1 or 2 only)

    Returns: A dictionary with the neuron's spines skeletons (points + structure arrays)
    """
    if num_spines != 1 and num_spines != 2:
        raise ValueError("Unsupported num_spines value: can only be 1 or 2")

    points = np.array(
        [
            [2, 2, 2, 0.1],  # spine 0 start (x, y, z, d)
            [2, 2, 4, 2.0],  # spine 0 end (x, y, z, d)
            [5, 5, 5, 0.1],  # spine 1 start (x, y, z, d)
            [5, 5, 9, 2.0],  # spine 1 end (x, y, z, d)
        ]
    )

    structure = np.array([[0, 2, -1], [2, 2, -1]])

    points = points[0:2] if num_spines == 1 else points
    structure = structure[0:1] if num_spines == 1 else structure

    coord_shift = np.array([neuron_idx, 0, 0, 0], dtype=np.float32)
    points = points + coord_shift

    spine_skeletons = {"points": points, "structure": structure}

    return spine_skeletons


def generate_spine_meshes(neuron_idx: int, num_spines: int) -> dict[str, np.ndarray]:
    """Generate multiple spine meshes whose coordinates are based on its neuron index.

    In order to make it simple, the same spine meshes are always created. The only difference is
    that the X axis is shifted by +neuron_idx units.
    Note: num_spines can only be 1 or 2
    Args:
        neuron_idx: The neuron's index
        num_spines: The number of spines of the neuron (valid values are 1 or 2 only)

    Returns: A dictionary with the neuron's spines meshes (offsets + points + vertices arrays)
    """
    if num_spines != 1 and num_spines != 2:
        raise ValueError("Unsupported num_spines value: can only be 1 or 2")

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

    triangles = triangles[0:6] if num_spines == 1 else triangles
    vertices = vertices[0:5] if num_spines == 1 else vertices

    coord_shift = np.array([neuron_idx, 0, 0], dtype=np.float32)
    vertices = vertices + coord_shift

    spine_meshes = {"offsets": offsets, "triangles": triangles, "vertices": vertices}

    return spine_meshes


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
    spine_skeletons = data[(neuron_name, "spine_skeletons")]
    spine_meshes = data[(neuron_name, "spine_meshes")]

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

        # Group /spines
        # Create top groups if they don't exist; create neuron ID group inside
        spines_grp = h5_file.require_group("spines")

        # Group /spines/meshes
        spines_meshes_grp = spines_grp.require_group("meshes")
        spines_id = spines_meshes_grp.create_group(neuron_name)

        spines_id.create_dataset("offsets", data=spine_meshes["offsets"])
        spines_id.create_dataset("triangles", data=spine_meshes["triangles"])
        spines_id.create_dataset("vertices", data=spine_meshes["vertices"])

        # Group /spines/skeletons
        spines_skel = spines_grp.require_group("skeletons")
        spines_skel_id = spines_skel.create_group(neuron_name)

        # Spine skeleton metadata
        spine_metadata = spines_skel_id.create_group("metadata")
        spine_metadata.attrs["cell_family"] = spine_morphology_family
        spine_metadata.attrs["version"] = spine_morphology_version

        spines_skel_id.create_dataset("points", data=spine_skeletons["points"])
        spines_skel_id.create_dataset("structure", data=spine_skeletons["structure"])

    print(f'Successfully added morphology-with-spines "{neuron_name}".')


def create_sample_data(
    output_file: str, num_neurons: int = 1, num_spines: int = 2, random_data: bool = False
):
    """Generate the sample data and write it to the given output file.

    Args:
        output_file: Filepath to output file that will be created
        num_neurons: Number of neurons to be written in the file
        num_spines: Number of spines (per neuron) to be written in the file
        random_data: Whether spine data must be generated randomly or not

    Returns: None
    """
    for i in range(num_neurons):
        neuron_name = f"neuron_{i}"
        if random_data:
            spine_table = generate_random_spine_data(neuron_name, num_spines)
        else:
            spine_table = generate_spine_data(neuron_name, num_spines)

        neuron_skeleton = generate_neuron_skeleton(i)
        soma_mesh = generate_soma_mesh(i)
        spine_skeletons = generate_spine_skeletons(i, num_spines)
        spine_meshes = generate_spine_meshes(i, num_spines)

        data = {
            (neuron_name, "spine_table"): spine_table,
            (neuron_name, "neuron_skeleton"): neuron_skeleton,
            (neuron_name, "soma_mesh"): soma_mesh,
            (neuron_name, "spine_skeletons"): spine_skeletons,
            (neuron_name, "spine_meshes"): spine_meshes,
        }

        write_neuron_data(output_file, neuron_name, data)

    print(f"{output_file} successfully created.")


def main() -> None:
    """Main function.

    Returns: None
    """
    parser = argparse.ArgumentParser(
        description="Generate sample data in morphology-with-spines format"
    )

    # Output file (string)
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # Number of neurons (int)
    parser.add_argument("-nneurons", type=int, default=1, help="Number of neurons")

    # Number of spines (int)
    parser.add_argument("-nspines", type=int, default=2, help="Number of spines per neuron")

    # Whether to generate random data (boolean)
    parser.add_argument("--random_data", action="store_true", help="Generate random data")

    args = parser.parse_args()

    print("Output file:", args.output)
    print("Number of neurons:", args.nneurons)
    print("Number of spines per neuron:", args.nspines)
    print("Random data enabled:", args.random_data)

    create_sample_data(args.output, args.nneurons, args.nspines, args.random_data)


if __name__ == "__main__":
    main()
