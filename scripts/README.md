# Scripts

Utility scripts for creating, converting, and processing morphology-with-spines data.


## Data creation

| Script | Description |
|--------|-------------|
| `create_sample_data.py` | Main entry point for generating sample morphology-with-spines H5 files. Supports configuring number of neurons, spines, grouping strategy, and centering. Run with `-h` for usage. |
| `create_sample_data_writer.py` | Helper module that writes the assembled data dictionary to an H5 file following the format specification. |
| `create_sample_morphology_data.py` | Helper module that generates synthetic neuron skeletons and soma meshes. |
| `create_sample_spines_data.py` | Helper module that generates synthetic spine skeletons, meshes, head/neck classification, and spine tables. |

### Example usage

```bash
# Generate a file with 2 neurons, 4 spines each, centered, grouped by neuron
python create_sample_data.py -o output.h5 -nneurons 2 -nspines 4 --by-neuron --centered
```


## Format conversion

| Script | Description |
|--------|-------------|
| `h5_dataframe_to_h5_datasets_group.py` | Converts legacy morphology-with-spines files (v0.1, spine table stored as pandas DataFrame) to the current format (v1.0, column-wise datasets). Required for files created before morph-spines v1.0. |

### Example usage

```bash
python h5_dataframe_to_h5_datasets_group.py input.h5 output.h5
```


## Mesh processing

| Script | Description |
|--------|-------------|
| `reduce_mesh_obj.py` | Reduces an OBJ morphology mesh to 10% of its original size using Open3D. |
| `reduce_soma_mesh.py` | Reduces the soma mesh inside a morphology-with-spines H5 file to 10% of its original size. |

These scripts require `open3d` which is not a dependency of the main package.
