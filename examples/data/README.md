# Morphology with spines file format documentation

This document describes the morphology with spines file format supported by this package.

The file is in HDF5 format and the information is structured into different groups and datasets,
described below. Some of the datasets are optional, as indicated. In order to reduce the file size,
some datasets are compressed, when indicated in their description.

Each file can contain the information of one or multiple neurons, together with multiple spines.

The spines data (skeletons and meshes) can be stored as a library, which can then be reused by
multiple neurons, so the same spine morphology can be instantiated multiple times and by multiple
neurons.


## `/edges` group

The `/edges` group contains one spines table for each neuron present in the file. Each spines table
is stored inside the neuron's name subgroup, and it describes different properties related to the 
neuron's spines.

For example, for a neuron named `"01234"`, the corresponding spines table will be stored under
`/edges/01234`.

The spines table is stored as an H5 group of datasets (v1.0 format). The metadata group contains
the version of the format.

The metadata is stored as a group under the neuron's subgroup called `metadata` and contains
the version (in major.minor format) of the spines table. The version is stored as an attribute of
the group with the name `version` and it is represented as an array of unsigned integers. The
presence of this group is mandatory.

Following the example above, the metadata would be stored as a group called
`/edges/01234/metadata` with a `version` attribute containing the array `[1, 0]`.

Note: files using the deprecated v0.1 format (pandas DataFrame) are no longer supported for
reading. Use the `h5_dataframe_to_h5_datasets_group.py` conversion script under the `scripts`
folder to migrate them.

### Spines table

The spines table contains information about the neuron spines. Each row represents a different
spine and each column describes a property. Therefore, the number of rows of the DataFrame equals
the number of spines belonging to the neuron.

There are 21 mandatory columns, as follows:

- `afferent_surface_x`: Follows SONATA specification: spine's position on the surface of a
  cylindrical cell segment, radially outward from the center position in the direction of the other
  cell in micrometers, X dimension (type: float)
- `afferent_surface_y`: Spine's afferent surface, Y dimension (type: float)
- `afferent_surface_z`: Spine's afferent surface, Z dimension (type: float)
- `afferent_center_x`: Follows SONATA specification: spine's position on the axis of the cell's
  section/segment in micrometers, X dimension (type: float)
- `afferent_center_y`: Spine's afferent center, Y dimension (type: float)
- `afferent_center_z`: Spine's afferent center, Z dimension (type: float)
- `spine_morphology`: A string used to locate the spine data inside the H5 file: spine meshes under
  `/spines/meshes` and spine skeletons under `/spines/skeletons` groups (type: string)
- `spine_id`: Spine index, in 0-based format, where spine data is located inside the
  `spine_morphology` datasets. Multiple rows can point at the same spine index (type: uint)
- `spine_length`: Length of the spine, from its root to its tip (type: float)
- `spine_orientation_vector_x`: Spine's normalized orientation vector, pointing at spine's tip from
  its root, X dimension (type: float)
- `spine_orientation_vector_y`: Spine's normalized orientation vector, Y dimension (type: float)
- `spine_orientation_vector_z`: Spine's normalized orientation vector, Z dimension (type: float)
- `spine_rotation_x`: Spine's quaternion rotation, to be applied to get the true orientation of the
  spine, X dimension (type: float)
- `spine_rotation_y`: Spine's rotation, Y dimension (type: float)
- `spine_rotation_z`: Spine's rotation, Z dimension (type: float)
- `spine_rotation_w`: Spine's rotation, W dimension (type: float)
- `afferent_section_id`: Follows SONATA specification: the specific section ID where a spine is
  placed (type: uint)
- `afferent_segment_id`: Follows SONATA specification: numerical index of the section of the cell
  where the spine is placed (type: int)
- `afferent_segment_offset`: Follows SONATA specification: spine's offset within the segment in
  micrometers (type: float)
- `afferent_section_pos`: Follows SONATA specification: spine's fractional position along the
  length of the section, normalized to the range [0, 1], where 0 is the start and 1 is  the end of
  the section (type: float)

Additionally, we can have the following columns as optional:

- `spine_volume`: Spine's head volume (type: float)
- `spine_neck_diameter`: Spine's neck diameter (type: float)
- `spine_type`: Spine morphological type classification (type: string). Valid values are:
  `undefined`, `thin`, `long_thin`, `mushroom`, `stubby`, `filopodium`, `branched`, 

The presence of the spines table is mandatory.

#### Spines table format

The spines table is stored column-wise, having one H5 dataset per column. The name of
the dataset corresponds to the name of the column.

All datasets must be stored under the same H5 group (the neuron name) and must have exactly
the same length. Datasets cannot be multidimensional: only 1-dimensional arrays and
scalars are accepted (in which case they are interpreted as an array of a single element).

The H5 group can only contain the datasets representing spines table columns and the mandatory
`metadata` subgroup. No other subgroups or datasets with different length are allowed.


## `/morphology` group

The `/morphology` group contains the morphology structure of each neuron present in the file. There
is a subgroup with the neuron ID for each neuron.

For example, for a neuron ID `"01234"`, the corresponding morphology structure will be stored under
`/morphology/01234`.

The contents of each subgroup follow the morphology structure described in 
[H5 v1](https://morphology-documentation.readthedocs.io/en/latest/h5v1.html). However, only the 
`metadata`, `/points` and `/structure` datasets are present. Following the example above, there
would be three datasets: `/morphology/01234/metadata`, `/morphology/01234/points` and
`/morphology/01234/structure`.

The presence of neuron morphology datasets (`/metadata`, `/points` and `/structure`) is mandatory.


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
triangle, and therefore, its visibility. To learn more, about this topic, see:
https://learnopengl.com/Advanced-OpenGL/Face-culling

The presence of soma mesh datasets (`/vertices` and `/triangles`) is optional.


## `/spines` group

The `/spines` group contains two subgroups called `/meshes` and `/skeletons`.

### `/spines/meshes` subgroup

The `/spines/meshes` subgroup contains the mesh of each spine present in the file. Spine meshes can
be divided into subgroups. The most intuitive way to organize them is by the neuron ID where they
belong to, although this is not a requirement. For example, they can also be organized as
libraries, subdivided by spine shapes or types. In any case, the `spine_morphology` entry in the
spines table must match the subgroup where meshes are stored and the `spine_id` entry must be
set accordingly to index the spine location within the datasets under `spine_morphology` group.

For example, if we split the spines by neuron ID, for a neuron ID `"01234"`, the corresponding 
spine meshes will be stored under `/spines/meshes/01234`.

The spine meshes are represented with three datasets: `/offsets`, `/triangles` and `/vertices`.
These three datasets are the only ones stored compressed in the HDF5 file.

The `/triangles` and `/vertices` datasets represent points and triangles in the 3D space in the
same way as the `/soma/meshes` are described.

The `/offsets` dataset is a list of pairs where each pair points at the first `vertex` and 
`triangle` of each spine respectively. For easiness, if there are `NS` spines, there will be `NS+1`
pairs of offsets. The first pair will always be `(0, 0)` and the last one will be `(NV+1, NT+1)` 
where `NV` is the number of vertices and `NT` is the number of triangles. The `spine_id` entry in
the spine table is used to index this list.

Therefore, to get the mesh for the spine with ID `IDS`, we can do the following:

```python
spine_vertices = neuron_vertices[neuron_offsets[IDS]:neuron_offsets[IDS+1]]
spine_triangles = neuron_triangles[neuron_offsets[IDS]:neuron_offsets[IDS+1]]
```

#### Head/neck triangle classification (optional)

The `/head_neck_values` dataset together with an additional column in the `/offsets` dataset
provide an optional classification of each spine's mesh triangles into head and neck regions.
Within each spine's triangle range, triangles are assumed to be sorted: all neck triangles first,
then all head triangles. For branched spines with multiple heads, each head's triangles are
grouped contiguously and appear in order (head 0, head 1, ...) after all the neck triangles.

This uses the same double-index approach as vertices and triangles: the `/head_neck_values`
dataset is a flat 1D array containing all head/neck offset values concatenated across all spines.
Column 2 of the `/offsets` dataset indexes into `/head_neck_values` per spine, so for spine `IDS`:

```python
hn_start = offsets[IDS, 2]
hn_end = offsets[IDS+1, 2]
spine_hn_offsets = head_neck_values[hn_start:hn_end]
```

The resulting `spine_hn_offsets` is an offset-style array of `H + 2` entries for a spine with `H`
heads, where all indices are local to the spine's own triangle range:

- Undefined triangles: `spine_triangles[0 : spine_hn_offsets[0]]`
- Neck triangles: `spine_triangles[spine_hn_offsets[0] : spine_hn_offsets[1]]`
- Head N triangles: `spine_triangles[spine_hn_offsets[N+1] : spine_hn_offsets[N+2]]`
- `spine_hn_offsets[-1]` equals the total number of triangles for that spine.

Undefined triangles are always returned regardless of filtering options. When all triangles are
classified (no undefined region), `spine_hn_offsets[0]` is `0`. When no triangles are classified
(all undefined), the spine has an empty slice in `head_neck_values` (equal consecutive offsets in
column 2) and all triangles are returned regardless of the filter flags.

For spines without head/neck classification, the consecutive offsets in column 2 are equal
(`offsets[IDS, 2] == offsets[IDS+1, 2]`), resulting in an empty slice. In this case, all
triangles are considered undefined and are always returned regardless of filtering options.

For backward compatibility with older files: if the `/head_neck_values` dataset is not present
or the `/offsets` dataset has only 2 columns (vertices, triangles), all triangles are considered
undefined.

The presence of spine meshes datasets (`vertices`, `/triangles`, `/offsets` and
`/head_neck_values`) is optional.

### `/spines/skeletons` subgroup

The `/spines/skeletons` subgroup contains the skeleton structure of each spine present in the file.
Similarly to the `/spines/meshes/...` subgroups, we can group spines by their neuron ID, or by
another criteria. All the skeletons of the same subgroup are grouped together in a single
structure. In any case, the `spine_morphology` entry in the spine table must match the subgroup
where skeletons are stored and the `spine_id` entry must be set accordingly to index the spine
location within the datasets under `spine_morphology` group.

For example, if we split the spines by neuron ID, for a neuron ID `"01234"`, the corresponding
spine skeletons will be stored under `/spines/skeletons/01234`.

The format of the skeletons complies to the aforementioned H5 v1 morphology structure: 
[H5 v1](https://morphology-documentation.readthedocs.io/en/latest/h5v1.html). However, only the 
`/metadata`, `/points` and `/structure` datasets are present. Following the example above, there
would be three datasets: `/spines/skeletons/01234/metadata`, `/spines/skeletons/01234/points` and
`/spines/skeletons/01234/structure`.

The presence of spine skeletons datasets (`/metadata`, `/points` and `/structure`) is mandatory.
