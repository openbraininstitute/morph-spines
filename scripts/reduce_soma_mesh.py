"""Read a morphology with spines file and reduce the soma mesh down to 10% of the original size."""

from pathlib import Path

import h5py
import numpy as np
import open3d as o3d

input_file = Path("./data/morphology_with_spines/morphology_with_spines.obj")
output_dir = Path(f"{input_file.parent}/output")
output_dir.mkdir(exist_ok=True)
output_file = Path(f"{output_dir}/{input_file.stem}_simplified{input_file.suffix}")

# Triangle scaling factor
scale_factor = 0.1

# Morphology ID for whom their soma meshes will be simplified (comma separated)
morph_ids = {"864691134884740346"}

# Read original file and write to new output file, simplifying only target meshes
with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:
    for name, _item in f_in.items():
        if name != "soma":
            # Copy other top-level groups/datasets
            f_in.copy(name, f_out)
        else:
            soma_in = f_in["soma"]
            soma_out = f_out.create_group("soma")

            # Copy all subgroups except meshes
            for sub_name, _sub_item in soma_in.items():
                if sub_name != "meshes":
                    soma_in.copy(sub_name, soma_out)

            # Process meshes
            meshes_in = soma_in["meshes"]
            meshes_out = soma_out.create_group("meshes")

            for mesh_id, mesh_item in meshes_in.items():
                if mesh_id not in morph_ids:
                    # Copy unchanged meshes
                    meshes_in.copy(mesh_id, meshes_out)
                else:
                    # Simplify this mesh
                    vertices = mesh_item["vertices"][:]
                    triangles = mesh_item["triangles"][:]

                    mesh = o3d.geometry.TriangleMesh()
                    mesh.vertices = o3d.utility.Vector3dVector(vertices)
                    mesh.triangles = o3d.utility.Vector3iVector(triangles)
                    mesh.compute_vertex_normals()

                    target_triangles = max(1, int(len(triangles) * scale_factor))
                    mesh_simpl = mesh.simplify_quadric_decimation(target_triangles)

                    vertices_simpl = np.asarray(mesh_simpl.vertices, dtype=np.float32)
                    triangles_simpl = np.asarray(mesh_simpl.triangles, dtype=np.int32)

                    # Save simplified mesh
                    mesh_group_new = meshes_out.create_group(mesh_id)
                    mesh_group_new.create_dataset("vertices", data=vertices_simpl)
                    mesh_group_new.create_dataset("triangles", data=triangles_simpl)

print("Selected meshes simplified and file size reduced")
