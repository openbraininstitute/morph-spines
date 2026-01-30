"""Auxiliary functions to create the spines data of a morphology-with-spines."""

from itertools import cycle

import numpy as np
from numpy.typing import NDArray

# Global variables
# Format versions
spine_table_version = np.array([1, 0], dtype=np.uint32)
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

mesh_library_shapes_and_offsets = {"tetrahedron": (4, 4), "pyramid": (5, 6), "prism": (6, 8)}


def generate_spines_table(
    neuron_skeleton: dict, neuron_coll_names: list[str], spines_collections: dict
) -> dict[str, NDArray]:
    """Create a 2D array representing the spines table for the given neuron.

    For the given neuron, all the spines from all collections found in neuron_coll_names will be
    added to the neuron. The spine location is based on the middle point of each neuron's sections.
    If there are more spines than neuron sections, spine locations will be repeated. This is a
    simple strategy to ensure all spines are touching the neuron, to be used only for testing.
    Assumption: all spines skeletons are centered at the origin and placed upright.

    Args:
        neuron_skeleton: Dictionary with the neuron skeleton (metadata + points + structure arrays)
        neuron_coll_names: Names of spine collections to be created for the neuron
        spines_collections: Dictionary with the spines collections (metadata + points + structure
        arrays)

    Returns: Spine table for the neuron in a 2-dimensional array
    """
    # Create the spine table as an empty structured array
    num_spines = 0
    for coll in neuron_coll_names:
        num_spines += len(spines_collections[coll]["structure"])

    data = np.empty(num_spines, dtype=dtypes)

    spine_idx = 0
    for coll_name in neuron_coll_names:
        spines_skeletons = spines_collections[coll_name]

        for spine_coll_idx in range(len(spines_skeletons["structure"])):
            # Pick neuron section (cycle if needed)
            sec_id = spine_idx % len(neuron_skeleton["structure"])

            # Place the spine at the middle point of the section
            sec_offset = neuron_skeleton["structure"][sec_id][0]
            next_sec_offset = (
                neuron_skeleton["structure"][sec_id + 1][0]
                if sec_id + 1 < len(neuron_skeleton["structure"])
                else len(neuron_skeleton["structure"])
            )
            sec_start = neuron_skeleton["points"][sec_offset][:3]
            sec_end = neuron_skeleton["points"][next_sec_offset - 1][:3]

            spine_start = 0.5 * (sec_start + sec_end)

            data[spine_idx]["afferent_surface_x"] = spine_start[0]
            data[spine_idx]["afferent_surface_y"] = spine_start[1]
            data[spine_idx]["afferent_surface_z"] = spine_start[2]

            data[spine_idx]["afferent_center_x"] = data[spine_idx]["afferent_surface_x"]
            data[spine_idx]["afferent_center_y"] = data[spine_idx]["afferent_surface_y"]
            data[spine_idx]["afferent_center_z"] = data[spine_idx]["afferent_surface_z"]

            data[spine_idx]["spine_id"] = spine_coll_idx
            data[spine_idx]["spine_morphology"] = coll_name

            # To simplify the calculation of the spine lenght, we take the first and last point of
            # the spine and compute the difference on the Y axis. This is not biologically correct,
            # it is just used for testing purposes
            spine_offset = spines_skeletons["structure"][spine_coll_idx][0]
            next_spine = (
                spines_skeletons["structure"][spine_coll_idx + 1][0]
                if spine_coll_idx + 1 < len(spines_skeletons["structure"])
                else len(spines_skeletons["structure"])
            )
            spine_points_start = spines_skeletons["points"][spine_offset][:3]
            spine_points_end = spines_skeletons["points"][next_spine - 1][:3]
            data[spine_idx]["spine_length"] = abs(spine_points_end[1] - spine_points_start[1])

            data[spine_idx]["spine_orientation_vector_x"] = 0.0
            data[spine_idx]["spine_orientation_vector_y"] = 1.0
            data[spine_idx]["spine_orientation_vector_z"] = 0.0

            data[spine_idx]["spine_rotation_x"] = 0.0
            data[spine_idx]["spine_rotation_y"] = 0.0
            data[spine_idx]["spine_rotation_z"] = 0.0
            data[spine_idx]["spine_rotation_w"] = 1.0

            data[spine_idx]["afferent_section_id"] = sec_id
            data[spine_idx]["afferent_segment_id"] = 0
            data[spine_idx]["afferent_segment_offset"] = 0
            data[spine_idx]["afferent_section_pos"] = 0

            spine_idx += 1

    spines_table = {
        "spine_table_version": spine_table_version,
        "spines_table_data": data,
    }
    return spines_table


def generate_spines_tables(
    neuron_skeletons: dict, spines_collections: dict, group_by_neuron: bool
) -> dict:
    """Generate the spines tables for the given sets of neurons and collections.

    If grouping by neuron: the list of neuron names must match the list of spine collection names.
    In this case, each neuron will have all the spines from its corresponding collection.
    If grouping by collection: all neurons will have all the spines from every collection.

    Args:
        neuron_skeletons: A dictionary with the neuron names and their skeletons
        spines_collections: A dictionary with the collection names and their spines skeletons
        group_by_neuron: Whether to group by neuron (true) or by collection (false)

    Returns: A dictionary with the neuron names and their spines tables
    """
    spines_tables = {}
    neuron_names = neuron_skeletons.keys()
    collection_names = spines_collections.keys()
    neuron_spines_collections = {}
    if group_by_neuron:
        if set(neuron_names) != set(collection_names):
            raise ValueError("Neuron names must match collection names when grouping by neuron")

        # If grouping by neuron, we only take the spines from the matching collection name
        for neuron_name in neuron_names:
            neuron_spines_collections[neuron_name] = [neuron_name]

    else:
        # If grouping by collection, we take all the spines from all collections for each neuron
        for neuron_name in neuron_names:
            neuron_spines_collections[neuron_name] = list(collection_names)

    for neuron_name, neuron_coll_names in neuron_spines_collections.items():
        spines_tables[neuron_name] = generate_spines_table(
            neuron_skeletons[neuron_name], neuron_coll_names, spines_collections
        )

    return spines_tables


def generate_spines_skeletons(num_spines: int) -> dict:
    """Generate multiple spine skeletons whose coordinates are based on its spine index.

    In order to make it simple, the same spine skeletons are always created. The only difference is
    that the spines are progressively longer along the Y axis (+0.1 units for each spine).

    Args:
        num_spines: The number of spines to be generated.

    Returns: A dictionary with the spines skeletons (metadata version + points + structure arrays)
    """
    points = []
    structure = []
    for spine_idx in range(num_spines):
        # Spine start: always centered at (0, 0, 0), in (x, y, z, diameter) format
        points.append([0.0, 0.0, 0.0, 0.1])
        # Spine end: spines are always placed upright, progressively growing along Y axis
        points.append([0.0, (spine_idx + 1) * 0.1, 0.0, 0.0])

        # Structure: all spines have 2 points (offset +2), the rest is always the same
        structure.append([spine_idx * 2, 2, -1])

    spines_skeletons = {
        ("metadata", "cell_family"): spine_morphology_family,
        ("metadata", "cell_version"): spine_morphology_version,
        "points": np.array(points, dtype=np.float32),
        "structure": np.array(structure, dtype=np.int32),
    }

    return spines_skeletons


def generate_all_spines_skeletons(coll_names: list, num_spines_per_col: int) -> dict:
    """Create the given number of spine skeletons for each given collection.

    Args:
        coll_names: List of collection names
        num_spines_per_col: Number of spines per collection

    Returns: A dictionary with the spine collection names and their spines skeletons, which in turn
    is another dictionary (metadata version + points + structure arrays)
    """
    all_spines_skeletons = {}
    for coll_name in coll_names:
        all_spines_skeletons[coll_name] = generate_spines_skeletons(num_spines_per_col)

    return all_spines_skeletons


def generate_spines_meshes(shape: str, spines_skeletons: dict[str, NDArray]) -> dict:
    """Generate multiple spine meshes from the given skeletons and with the given shape.

    In order to make it simple, only the start and end point of each spine is considered. The
    created mesh is based on a mesh library with tetrahedron, pyramid and prism shapes.

    Args:
        shape: A string describing the shape of the mesh. Possible values are: tetrahedron, pyramid
        and prism
        spines_skeletons: A dictionary containing the spines skeletons (points and structure arrays)

    Returns: A dictionary with the spines meshes (offsets + points + vertices arrays)
    """
    points = spines_skeletons["points"]
    structure = spines_skeletons["structure"]
    num_spines = len(structure)

    all_vertices = []
    # Offset format: [vertices_offset, triangles_offset]
    all_offsets = [[0, 0]]
    # Shape = {geometry: (vertices_offset, triangles_offset)}
    shape_offsets = mesh_library_shapes_and_offsets[shape]

    for spine_idx in range(num_spines):
        first_pt = structure[spine_idx, 0]
        next_pt = structure[spine_idx + 1, 0] if spine_idx + 1 < num_spines else len(points)

        spine_pts = points[first_pt:next_pt]

        if len(spine_pts) < 2:
            # Single-point spine: we consider this point as the center of the spine and create an
            # arbitrary shape around it with length = 2 * radius along Y axis
            spine_start_point = [
                spine_pts[0][0],
                spine_pts[0][1] + spine_pts[0][3],
                spine_pts[0][2],
                spine_pts[0][3],
            ]
            spine_end_point = [
                spine_pts[0][0],
                spine_pts[0][1] - spine_pts[0][3],
                spine_pts[0][2],
                spine_pts[0][3],
            ]
        else:
            # Spine with 2 or more points: consider only first and last points
            spine_start_point = spine_pts[0]
            spine_end_point = spine_pts[-1]

        # Simplification: assume spines are centered and upright, compute length based on Y values
        spine_length = abs(spine_end_point[1] - spine_start_point[1])

        mesh_shift = np.array(meshes_library[(shape, "shift")], dtype=np.float64) * spine_length
        mesh_points = np.array(meshes_library[(shape, "points")], dtype=np.float64) + mesh_shift
        all_vertices.append(mesh_points)
        all_offsets.append([shape_offsets[0] * (spine_idx + 1), shape_offsets[1] * (spine_idx + 1)])

    spines_meshes = {
        "offsets": np.vstack(all_offsets),
        "triangles": np.tile(
            np.array(meshes_library[(shape, "triangles")], dtype=np.int32), (num_spines, 1)
        ),
        "vertices": np.vstack(all_vertices),
    }

    return spines_meshes


def generate_all_spines_meshes(spines_skeletons: dict[str, dict[str, NDArray]]) -> dict:
    """Create a spine mesh for each spine skeleton.

    Args:
        spines_skeletons: Dictionary with the spines collections names and their skeletons

    Returns: A dictionary with the spine collection names and their spines skeletons, which in turn
    is another dictionary (metadata version + points + structure arrays)
    """
    all_spines_meshes = {}
    coll_name_shape = list(
        zip(spines_skeletons.keys(), cycle(mesh_library_shapes_and_offsets.keys()))
    )

    for coll_name, shape in coll_name_shape:
        all_spines_meshes[coll_name] = generate_spines_meshes(shape, spines_skeletons[coll_name])

    return all_spines_meshes
