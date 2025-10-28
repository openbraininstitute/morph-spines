# Given an .obj mesh file, reduce the given mesh down to 10% of the original size

import open3d as o3d

input_file = "./data/morphology_meshes/864691134884740346.obj"
base, ext = input_file.rsplit(".", 1)
output_file = base + "_1." + ext

# Triangle scaling factor
scale_factor = 0.1

# Load the mesh
mesh = o3d.io.read_triangle_mesh(input_file)
print(f"Original mesh: {len(mesh.vertices)} vertices, {len(mesh.triangles)} faces")

# Ensure normals and watertightness
mesh.remove_duplicated_vertices()
mesh.remove_degenerate_triangles()
mesh.remove_duplicated_triangles()
mesh.remove_non_manifold_edges()
mesh.compute_vertex_normals()

# Simplify the mesh
target_faces = int(len(mesh.triangles) * scale_factor)

simplified = mesh.simplify_quadric_decimation(target_faces)
simplified.compute_vertex_normals()

print(f"Simplified mesh: {len(simplified.vertices)} vertices, {len(simplified.triangles)} faces")

# Save the result
output_path = "./data/morphology_meshes/neuron_simplified.obj"
o3d.io.write_triangle_mesh(output_path, simplified)
print(f"Simplified mesh saved to {output_path}")

# If in an ipython notebook, show the mesh
#simplified.show()

