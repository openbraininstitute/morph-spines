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

# Spine table format versions
SPINE_TABLE_VER_PANDAS_DF = (0, 1)  # Deprecated: pandas DataFrame format
SPINE_TABLE_VER_H5_DATASETS = (1, 0)  # Current: column-wise HDF5 datasets

# Mandatory columns in the spine table and their expected dtype kinds:
#   "f" = floating point, "i" = signed integer, "ui" = unsigned integer, "str" = string
MANDATORY_COLUMNS = {
    "afferent_surface_x": "f",
    "afferent_surface_y": "f",
    "afferent_surface_z": "f",
    "afferent_center_x": "f",
    "afferent_center_y": "f",
    "afferent_center_z": "f",
    "spine_morphology": "str",
    "spine_id": "ui",
    "spine_length": "f",
    "spine_orientation_vector_x": "f",
    "spine_orientation_vector_y": "f",
    "spine_orientation_vector_z": "f",
    "spine_rotation_x": "f",
    "spine_rotation_y": "f",
    "spine_rotation_z": "f",
    "spine_rotation_w": "f",
    "afferent_section_id": "ui",
    "afferent_segment_id": "i",
    "afferent_segment_offset": "f",
    "afferent_section_pos": "f",
}

# Optional columns and their expected dtype kinds
OPTIONAL_COLUMNS = {
    "spine_volume": "f",
    "spine_neck_diameter": "f",
    "spine_type": "str",
}
