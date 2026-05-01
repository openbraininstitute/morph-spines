"""Defines the schema used in the h5 file morphology-with-spines format."""

# Names of groups in the morphology-w-spines hdf5 file
# Root groups
GRP_EDGES = "edges"
GRP_MORPH = "morphology"
GRP_SOMA = "soma"
GRP_SPINES = "spines"
# Sub-groups
GRP_MESHES = "meshes"
GRP_SKELETONS = "skeletons"
GRP_METADATA = "metadata"
# Sub-sub-groups inside meshes
GRP_OFFSETS = "offsets"
GRP_TRIANGLES = "triangles"
GRP_VERTICES = "vertices"
GRP_HEAD_NECK_VALUES = "head_neck_values"

# Column indices within the offsets dataset
OFF_COL_VERTICES = 0
OFF_COL_TRIANGLES = 1
OFF_COL_HEAD_NECK = 2

# Columns of edge table dataframes
COL_SPINE_MORPH = "spine_morphology"
COL_SPINE_ID = "spine_id"
COL_SPINE_TYPE = "spine_type"
COL_ROTATION = ["spine_rotation_x", "spine_rotation_y", "spine_rotation_z", "spine_rotation_w"]
COL_TRANSLATION = ["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]
COL_AFF_SEC = "afferent_section_id"

# Metadata attributes
ATT_VERSION = "version"
