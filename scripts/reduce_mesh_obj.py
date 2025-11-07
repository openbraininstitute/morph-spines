"""Given an .obj mesh file, reduce the given mesh down to 10% of the original size."""

from pathlib import Path

import open3d as o3d

input_file = Path("./data/morphology_meshes/morphology_mesh.obj")
output_dir = Path(f"{input_file.parent}/output")
output_dir.mkdir(exist_ok=True)
output_file = Path(f"{output_dir}/{input_file.stem}_simplified{input_file.suffix}")

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
o3d.io.write_triangle_mesh(output_file, simplified)
print(f"Simplified mesh saved to {output_file}")

# If in an ipython notebook, show the mesh
# simplified.show()
