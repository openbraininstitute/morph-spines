# Morphology with spines file format documentation

This document describes the morphology with spines file format supported by this package.

The file is in HDF5 format and the information is structured into different groups and datasets,
described below.

Each file can contain the information of one or multiple neurons.

## `/edges` group

The `/edges` group contains one Pandas DataFrame for each neuron present in the file. Each
DataFrame is represented by a dataset inside the neuron's ID subgroup, and it describes different
properties related to the neuron's spines.

For example, for a neuron ID `"01234"`, the corresponding DataFrame will be stored under
`/edges/01234`.

The DataFrame can be read and written through Panda's `pandas.DataFrame.read_hdf()` and 
`pandas.DataFrame.to_hdf()` respectively.

### Spines' DataFrame

The spines DataFrame contains information about the neuron spines. Each row represents a different
spine and each column describes a property. Therefore, the number of rows of the DataFrame equals
the number of spines belonging to the neuron.

There are 20 columns, as follows:
- `afferent_surface_x`: Spine's afferent surface, X dimension (type: float64)
- `afferent_surface_y`: Spine's afferent surface, Y dimension (type: float64)
- `afferent_surface_z`: Spine's afferent surface, Z dimension (type: float64)
- `afferent_center_x`: Spine's afferent center, X dimension (type: float64)
- `afferent_center_y`: Spine's afferent center, Y dimension (type: float64)
- `afferent_center_z`: Spine's afferent center, Z dimension (type: float64)
- `spine_id`: Spine ID, matches its row position in 0-based format (type: int64)
- `spine_morphology`: Neuron ID which the spine belongs to (type: string)
- `spine_length`: Length of the spine (type: float64)
- `spine_orientation_vector_x`: Spine's orientation vector, X dimension (type: float64)
- `spine_orientation_vector_y`: Spine's orientation vector, Y dimension (type: float64)
- `spine_orientation_vector_z`: Spine's orientation vector, Z dimension (type: float64)
- `spine_rotation_x`: Spine's rotation, X dimension (type: float64)
- `spine_rotation_y`: Spine's rotation, Y dimension (type: float64)
- `spine_rotation_z`: Spine's rotation, Z dimension (type: float64)
- `spine_rotation_w`: Spine's rotation, W dimension (type: float64)
- `afferent_section_id`: Spine's afferent section ID (type: int64)
- `afferent_segment_id`: Spine's afferent segment ID (type: int64)
- `afferent_segment_offset`: Spine's afferent segment offset (type: float64)
- `afferent_section_pos`: Spine's afferent section position (type: float64)


## `/morphology` group

The `/morphology` group contains the morphology structure of each neuron present in the file. There
is a subgroup with the neuron ID for each neuron.

For example, for a neuron ID `"01234"`, the corresponding morphology structure will be stored under
`/morphology/01234`.

The contents of each subgroup follow the morphology structure described in 
[H5 v1](https://morphology-documentation.readthedocs.io/en/latest/h5v1.html). However, only the 
`/points` and `/structure` datasets are present. Following the example above, there would be two 
datasets: `/morphology/01234/points` and `/morphology/01234/structure`.


## `/soma` group

The `/soma` group contains a single subgroup called `/meshes`.

### `/soma/meshes` subgroup

The `/soma/meshes` subgroup contains the soma mesh of each neuron present in the file. There is a 
subgroup with the neuron ID for each neuron.

For example, for a neuron ID `"01234"`, the corresponding soma mesh will be stored under
`/soma/meshes/01234`.

The mesh is represented as two separate datasets: `/triangles` and `/vertices`.

The `/vertices` dataset contains an array of triplets, where each triplet represents a point in 3D 
coordinates (X, Y, Z). Each point represents a triangle's vertex in space.

The `/triangles` dataset contains an array of triplets, where each triplet references the 
`/vertices` dataset to describe a triangle by indexing their vertices.

For example, we can define a list of vertices in `[x, y, z]` coordinates as:
`[[0, 0, 0], [2, 0, 0], [2, 1, 0]]`

Then, to form a triangle with these three vertices, the triangle list would be:
`[[0, 1, 2]]`

Note: the vertex order (winding) matters, as this will determine the front and the back face of the 
triangle. To learn more, about this topic, see: https://learnopengl.com/Advanced-OpenGL/Face-culling


## `/spines` group

The `/spines` group contains two subgroups called `/meshes` and `/skeletons`.

### `/spines/meshes` subgroup

The `/spines/meshes` subgroup contains the mesh of each spine of each neuron present in the file. 
There is a subgroup with the neuron ID for each neuron where all the spine meshes are grouped 
together.

For example, for a neuron ID `"01234"`, the corresponding spine meshes will be stored under
`/spines/meshes/01234`.

The spine meshes are represented with three datasets: `/offsets`, `/triangles` and `/vertices`.

The `/triangles` and `/vertices` datasets represent points and triangles in the 3D space in the
same way as the `/soma/meshes` are described.

The `/offsets` dataset is a list of pairs where each pair points at the first `vertex` and 
`triangle` of each spine respectively. For easiness, if there are `NS` spines, there will be `NS+1`
pairs of offsets. The first pair will always be `(0, 0)` and the last one will be `(NV+1, NT+1)` 
where `NV` is the number of vertices and `NT` is the number of triangles.

Therefore, to get the mesh for the spine with ID `IDS`, we can do the following:

```python
spine_vertices = neuron_vertices[neuron_offsets[IDS]:neuron_offsets[IDS+1]]
spine_triangles = neuron_triangles[neuron_offsets[IDS]:neuron_offsets[IDS+1]]
```

### `/spines/skeletons` subgroup

The `/spines/skeletons` subgroup contains the skeleton structure of each spine of each neuron 
present in the file. There is a subgroup with the neuron ID for each neuron where all the spine 
skeleton structures are grouped together.

For example, for a neuron ID `"01234"`, the corresponding spine skeletons will be stored under
`/spines/skeletons/01234`.

The format of the skeletons complies to the aforementioned H5 v1 morphology structure: 
[H5 v1](https://morphology-documentation.readthedocs.io/en/latest/h5v1.html). However, only the 
`/points` and `/structure` datasets are present. Following the example above, there would be two 
datasets: `/spines/skeletons/01234/points` and `/spines/skeletons/01234/structure`.
