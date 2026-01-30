"""Auxiliary functions to write data in morphology-with-spines file format."""

import os
from pathlib import Path

import h5py
from trimesh import Trimesh


def write_neuron_data(output_file: str, data: dict) -> None:
    """Write the collection of neuron and spine data to file.

    Args:
        output_file: Filepath to output file that will be created
        data: dictionary with H5 groups/datasets to be created in the file

    Returns: None
    """
    mode = "a" if os.path.exists(output_file) else "w"
    with h5py.File(output_file, mode) as h5_file:
        # Group /edges
        # Get top group if it exists or create it otherwise
        edges_grp = h5_file.require_group("edges")
        for neuron_name, spine_table in data["spine_tables"].items():
            # Neuron subgroup must not exist, as neuron IDs should be unique
            spine_table_grp_name = str(f"/edges/{neuron_name}")
            spine_table_grp = edges_grp.create_group(spine_table_grp_name)

            # Spine table metadata
            edges_metadata = spine_table_grp.create_group("metadata")
            edges_metadata.attrs["version"] = spine_table["spine_table_version"]

            # Create as many datasets as columns in the spine table
            spine_table_data = spine_table["spines_table_data"]
            if spine_table_data.dtype.names is not None:
                # We know it's not None, just making mypy happy
                for col_name in spine_table_data.dtype.names:
                    dset_name = str(f"{spine_table_grp_name}/{col_name}")
                    spine_table_grp.create_dataset(dset_name, data=spine_table_data[col_name])

        # Group /morphology
        # Get top group if it exists or create it otherwise
        morph_grp = h5_file.require_group("morphology")
        for neuron_name, neuron_skeleton in data["neuron_skeletons"].items():
            morph_id = morph_grp.create_group(neuron_name)

            # Morphology metadata
            morph_metadata = morph_id.create_group("metadata")
            morph_metadata.attrs["cell_family"] = neuron_skeleton[("metadata", "cell_family")]
            morph_metadata.attrs["version"] = neuron_skeleton[("metadata", "cell_version")]

            morph_id.create_dataset("points", data=neuron_skeleton["points"])
            morph_id.create_dataset("structure", data=neuron_skeleton["structure"])

        # Group /soma/meshes
        # Get top groups if they exist or create them otherwise
        soma_grp = h5_file.require_group("soma")
        soma_meshes_grp = soma_grp.require_group("meshes")
        for neuron_name, soma_mesh in data["soma_meshes"].items():
            neuron_name_grp = soma_meshes_grp.create_group(neuron_name)
            neuron_name_grp.create_dataset("triangles", data=soma_mesh["triangles"])
            neuron_name_grp.create_dataset("vertices", data=soma_mesh["vertices"])

        # Group /spines
        # Get top groups if they exist or create them otherwise
        spines_grp = h5_file.require_group("spines")

        # Top group /spines/meshes
        spines_meshes_grp = spines_grp.require_group("meshes")

        for spine_col_name, spines_meshes in data["spines_meshes"].items():
            col_meshes_group = spines_meshes_grp.create_group(spine_col_name)
            col_meshes_group.create_dataset("offsets", data=spines_meshes["offsets"])
            col_meshes_group.create_dataset("triangles", data=spines_meshes["triangles"])
            col_meshes_group.create_dataset("vertices", data=spines_meshes["vertices"])

        # Top group /spines/skeletons
        spines_skel = spines_grp.require_group("skeletons")

        for spine_col_name, spines_skeletons in data["spines_skeletons"].items():
            col_skel_group = spines_skel.create_group(spine_col_name)

            # Spine skeleton metadata
            spine_metadata = col_skel_group.create_group("metadata")
            spine_metadata.attrs["cell_family"] = spines_skeletons[("metadata", "cell_family")]
            spine_metadata.attrs["version"] = spines_skeletons[("metadata", "cell_version")]

            col_skel_group.create_dataset("points", data=spines_skeletons["points"])
            col_skel_group.create_dataset("structure", data=spines_skeletons["structure"])

    print(f'Successfully written file "{output_file}".')


def write_neuron_meshes(output_file: str, neuron_meshes: dict[str, Trimesh]) -> None:
    """Write neuron meshes to file in OBJ format.

    The output_file parameter is used to prefix each mesh OBJ file, together with its neuron name.

    Args:
        output_file: Path of the morphology with spines file where neurons are stored.
        neuron_meshes: Dictionary of neuron meshes.

    Returns: None
    """
    data_file = Path(output_file).resolve()
    parent_dir = data_file.parent
    prefix = data_file.stem

    for neuron_name, neuron_mesh in neuron_meshes.items():
        mesh_file = parent_dir / f"{prefix}_{neuron_name}.obj"
        neuron_mesh.export(mesh_file)
