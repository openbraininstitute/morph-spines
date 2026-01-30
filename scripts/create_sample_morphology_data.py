"""Auxiliary functions to create the neuron morphology of a morphology-with-spines."""

import numpy as np
import trimesh
from morphio import SectionType
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as R

# Global variables
# Format versions
neuron_morphology_version = np.array([1, 3], dtype=np.uint32)
neuron_morphology_family = np.array([0], dtype=np.uint32)

# Morphology section definitions
SOMA_TYPE = SectionType.soma
AXON_TYPE = SectionType.axon


def generate_neuron_skeleton(neuron_idx: int) -> dict:
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

    morph_structure = np.array(
        [[0, SOMA_TYPE, -1], [2, AXON_TYPE, 0], [4, AXON_TYPE, 1], [6, AXON_TYPE, 1]],
        dtype=np.int32,
    )

    coord_shift = np.array([neuron_idx, 0, 0, 0], dtype=np.float32)
    morph_points = (morph_points + coord_shift).astype(np.float32)

    neuron_skeleton = {
        ("metadata", "cell_family"): neuron_morphology_family,
        ("metadata", "cell_version"): neuron_morphology_version,
        "points": morph_points,
        "structure": morph_structure,
    }

    return neuron_skeleton


def generate_neuron_skeletons(neuron_names: list[str]) -> dict:
    """Generate as many neuron skeletons as requested.

    In order to make it simple, the same morphology is always created. The only difference is that
    the X axis is shifted increasingly for each neuron.

    Args:
        neuron_names: Names of the neurons to be created

        Returns: A dictionary with all neuron names and their skeletons, which in turn is another
        dictionary (metadata version + points + structure arrays)
    """
    skeletons = {}
    for i, neuron_name in enumerate(neuron_names):
        skeletons[neuron_name] = generate_neuron_skeleton(i)

    return skeletons


def get_soma_points_from_skeleton(neuron_skeleton: dict[str, NDArray]) -> NDArray:
    """Given a neuron skeleton, extract its soma points.

    Args:
        neuron_skeleton: The neuron skeleton, with its points and structure arrays as a dictionary

    Returns: An array with all the soma points. If no soma points were found, returns a single soma
    point starting at the first neurite found
    """
    points = neuron_skeleton["points"]
    structure = neuron_skeleton["structure"]
    num_points = len(points)
    num_sections = len(structure)
    soma_points_list = []

    for sec_id in range(num_sections):
        sec_type = SectionType(structure[sec_id][1])

        # All soma points are at the beginning; break the loop when type is not SOMA anymore
        if sec_type != SOMA_TYPE:
            break

        first_pt = structure[sec_id, 0]
        last_pt = structure[sec_id + 1, 0] if sec_id + 1 < num_sections else num_points

        soma_points_list.append(points[first_pt:last_pt])

    if soma_points_list:
        soma_points = np.vstack(soma_points_list)
    else:
        # If no soma found, we create a small soma at the start of the first neurite
        first_pt = structure[0, 0]
        soma_points = np.array([[*points[first_pt, :3], 0.1]])

    return soma_points


def generate_sphere(center: NDArray, radius: float) -> trimesh.Trimesh:
    """Returns a sphere mesh, e.g.: to represent the soma.

    Args:
        center: central point of the sphere, in 3D coordinates
        radius: radius of the sphere

    Returns: a sphere mesh at the given center coordinates with the given radius
    """
    sphere = trimesh.creation.icosphere(subdivisions=3, radius=radius)
    sphere.apply_translation(center)
    return sphere


def generate_soma_mesh(soma_points: NDArray) -> trimesh.Trimesh:
    """Generate a spherical mesh to represent the neuron soma.

    This function is meant for testing purposes only and not for neuron visualization.

    Args:
        soma_points: Array with one or more soma points in the format (X, Y, Z, diameter)

    Returns: A Trimesh sphere object representing the neuron soma
    """
    if len(soma_points) == 0:
        raise ValueError("Soma points array cannot be empty")

    if len(soma_points) == 1:
        soma_mesh = generate_sphere(soma_points[:3], radius=soma_points[3])
    else:
        soma_center = np.mean(soma_points, axis=0)
        distances = np.linalg.norm(soma_points - soma_center, axis=1)
        soma_radius = np.median(distances)
        soma_mesh = generate_sphere(soma_center[:3], radius=soma_radius)

    return soma_mesh


def align_vectors(v_from: NDArray[np.float64], v_to: NDArray[np.float64]) -> R:
    """Compute a rotation that aligns one 3D vector to another.

    Args:
        v_from: 3-element vector representing the source direction.
        v_to: 3-element vector representing the target direction.

    Returns: A scipy Rotation object representing the alignment rotation.
    """
    v_from = np.asarray(v_from, dtype=np.float64)
    v_to = np.asarray(v_to, dtype=np.float64)

    n_from = np.linalg.norm(v_from)
    n_to = np.linalg.norm(v_to)
    if n_from == 0 or n_to == 0:
        raise ValueError("Input vectors must be non-zero")

    v_from /= n_from
    v_to /= n_to

    # SciPy expects lists of vectors
    rotation, _ = R.align_vectors([v_to], [v_from])
    return rotation


def generate_section_cylinder(
    p1: NDArray[np.float64],
    p2: NDArray[np.float64],
    radius: float = 0.1,
    sections: int = 16,
) -> trimesh.Trimesh | None:
    """Create a cylinder mesh between p1 and p2 points.

    The cylinder axis is aligned with the vector (p2 - p1), and the cylinder
    spans exactly between the two points.

    Args:
        p1: Starting point, in 3D space
        p2: Ending point, in 3D space
        radius: Radius of the cylinder
        sections: Number of radial sections of the cylinder mesh

    Returns: A Trimesh cylinder mesh, or None if p1 == p2.
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)

    direction = p2 - p1
    length = np.linalg.norm(direction)
    if length == 0:
        return None

    direction /= length

    # Create cylinder aligned with +Z, centered at origin
    cylinder = trimesh.creation.cylinder(
        radius=radius,
        height=length,
        sections=sections,
    )

    # Move cylinder so its base starts at Z=0
    cylinder.apply_translation([0.0, 0.0, length / 2.0])

    # Compute rotation from Z-axis to direction
    rot = align_vectors(np.array([0.0, 0.0, 1.0]), direction)

    # Build homogeneous transform
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = 0.5 * (p1 + p2)

    # Apply transform
    cylinder.apply_transform(T)

    return cylinder


def generate_neuron_mesh(neuron_skeleton: dict[str, NDArray]) -> trimesh.Trimesh:
    """Generate a very simplistic mesh of a neuron skeleton made of cylinders.

    This function is meant for testing purposes only and not for neuron visualization.

    Args:
        neuron_skeleton: The neuron skeleton, with its points and structure arrays as a dictionary

    Returns: A Trimesh object representing the neuron in a very simplistic way
    """
    points = neuron_skeleton["points"]
    structure = neuron_skeleton["structure"]
    num_points = len(points)
    num_sections = len(structure)
    section_meshes = []

    # Process soma first, which is found at the beginning of the array
    soma_points = get_soma_points_from_skeleton(neuron_skeleton)
    soma_mesh = generate_soma_mesh(soma_points)
    section_meshes.append(soma_mesh)

    # Then, process neurite sections
    for sec_id in range(num_sections):
        sec_type = SectionType(structure[sec_id][1])
        if sec_type == SOMA_TYPE:
            continue

        first_pt = structure[sec_id, 0]
        last_pt = structure[sec_id + 1, 0] if sec_id + 1 < num_sections else num_points

        section_pts = points[first_pt:last_pt]
        coords = section_pts[:, :3]
        radii = section_pts[:, 3] / 2

        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]

            if np.allclose(p1, p2):
                continue

            radius = 0.5 * (radii[i] + radii[i + 1])

            cylinder = generate_section_cylinder(p1, p2, radius=radius)
            if cylinder is not None:
                section_meshes.append(cylinder)

    whole_neuron_mesh = trimesh.util.concatenate(section_meshes)

    return whole_neuron_mesh


def generate_neuron_meshes(morph_skeletons: dict) -> dict[str, trimesh.Trimesh]:
    """Generate simplistic neuron meshes for all the given neuron skeletons.

    Args:
        morph_skeletons: Dictionary with neuron names and skeletons

    Returns: A dictionary with neuron names and their meshes
    """
    morph_meshes = {}
    for morph_name, morph_skeleton in morph_skeletons.items():
        morph_meshes[morph_name] = generate_neuron_mesh(morph_skeleton)

    return morph_meshes


def generate_soma_mesh_arrays(neuron_skeleton: dict[str, NDArray]) -> dict[str, NDArray]:
    """Generate a very simplistic mesh of the neuron soma in a spherical shape.

    This function is meant for testing purposes only and not for neuron visualization.

    Args:
        neuron_skeleton: The neuron skeleton, with its points and structure arrays as a dictionary

    Returns: A dictionary with vertices and triangles arrays representing the mesh
    """
    soma_points = get_soma_points_from_skeleton(neuron_skeleton)
    soma_mesh = generate_soma_mesh(soma_points)

    return {"triangles": soma_mesh.faces, "vertices": soma_mesh.vertices}


def generate_soma_meshes_arrays(neuron_skeletons: dict) -> dict[str, dict[str, NDArray]]:
    """Generate a very simplistic mesh for all the neuron skeletons.

    Args:
        neuron_skeletons: A dictionary with all the neuron skeletons and their names

    Returns: A dictionary with all neuron names and their soma mesh, which in turn is another
        dictionary (vertices + triangles arrays)

    """
    soma_meshes = {}
    for neuron_name, neuron_skeleton in neuron_skeletons.items():
        soma_meshes[neuron_name] = generate_soma_mesh_arrays(neuron_skeleton)

    return soma_meshes
