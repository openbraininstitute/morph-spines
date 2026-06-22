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

# Metadata attributes
ATT_VERSION = "version"

# Spine table format versions
SPINE_TABLE_VER_H5_DATASETS = (1, 0)  # Current: column-wise HDF5 datasets

# Columns of edge table dataframes (order follows documentation)
COL_TRANSLATION = [
    "afferent_surface_x",
    "afferent_surface_y",
    "afferent_surface_z",
]
COL_CENTER = [
    "afferent_center_x",
    "afferent_center_y",
    "afferent_center_z",
]
COL_SPINE_MORPH = "spine_morphology"
COL_SPINE_ID = "spine_id"
COL_SPINE_LENGTH = "spine_length"
COL_ORIENTATION = [
    "spine_orientation_vector_x",
    "spine_orientation_vector_y",
    "spine_orientation_vector_z",
]
COL_ROTATION = [
    "spine_rotation_x",
    "spine_rotation_y",
    "spine_rotation_z",
    "spine_rotation_w",
]
COL_AFF_SEC = "afferent_section_id"
COL_AFF_SEG_ID = "afferent_segment_id"
COL_AFF_SEG_OFFSET = "afferent_segment_offset"
COL_AFF_SEC_POS = "afferent_section_pos"
# Optional columns
COL_SPINE_VOLUME = "spine_volume"
COL_SPINE_NECK_DIAMETER = "spine_neck_diameter"
COL_SPINE_TYPE = "spine_type"

# Mandatory columns in the spine table and their expected dtype kinds:
#   "f" = floating point, "i" = signed integer, "ui" = unsigned integer,
#   "str" = string
MANDATORY_COLUMNS = {
    COL_TRANSLATION[0]: "f",
    COL_TRANSLATION[1]: "f",
    COL_TRANSLATION[2]: "f",
    COL_CENTER[0]: "f",
    COL_CENTER[1]: "f",
    COL_CENTER[2]: "f",
    COL_SPINE_MORPH: "str",
    COL_SPINE_ID: "ui",
    COL_SPINE_LENGTH: "f",
    COL_ORIENTATION[0]: "f",
    COL_ORIENTATION[1]: "f",
    COL_ORIENTATION[2]: "f",
    COL_ROTATION[0]: "f",
    COL_ROTATION[1]: "f",
    COL_ROTATION[2]: "f",
    COL_ROTATION[3]: "f",
    COL_AFF_SEC: "ui",
    COL_AFF_SEG_ID: "i",
    COL_AFF_SEG_OFFSET: "f",
    COL_AFF_SEC_POS: "f",
}

# Optional columns and their expected dtype kinds
OPTIONAL_COLUMNS = {
    COL_SPINE_VOLUME: "f",
    COL_SPINE_NECK_DIAMETER: "f",
    COL_SPINE_TYPE: "str",
}
