# morph-spines

A Python library for loading, writing, and accessing neuron morphologies with dendritic spine data
from HDF5 files. It provides structured access to spine skeletons, meshes, and spatial
transformations.


## Quick example

### Loading

```python
from morph_spines import load_morphology_with_spines

m = load_morphology_with_spines("neuron.h5", spines_are_centered=True, load_meshes=True)

# Access spine meshes
mesh = m.spines.spine_mesh(0)
print(mesh.vertices.shape, mesh.faces.shape)

# Get only the head region of a spine
head_mesh = m.spines.spine_mesh(0, include_neck=False)

# Spine type classification
spine_type = m.spines.spine_type(0)
```

### Writing

```python
from morph_spines import write_spine_table, write_morphology, write_soma_mesh

# Write a spine table (pandas DataFrame with mandatory columns)
write_spine_table("output.h5", "neuron_01", spine_table_df)

# Write neuron morphology skeleton
write_morphology("output.h5", "neuron_01", points, structure)

# Write soma mesh
write_soma_mesh("output.h5", "neuron_01", vertices, triangles)
```

### Validation

```python
from morph_spines import validate_morph_with_spines_file

# Check file structure only (groups, datasets, metadata)
result = validate_morph_with_spines_file("neuron.h5")

# Also check data integrity (shapes, dtypes, value ranges, cross-references)
result = validate_morph_with_spines_file("neuron.h5", check_data_integrity=True)

print(result)          # Human-readable summary
assert result.is_valid # Use programmatically
```


## Installation

```bash
pip install morph-spines
```

For development:

```bash
git clone https://github.com/openbraininstitute/morph-spines.git
cd morph-spines
pip install -e ".[test]"
```


## Features

- Load and write neuron morphologies with spine data from/to HDF5 files
- Access the spine table with per-spine properties (position, orientation, section placement)
- Access spine skeletons (via NeuroM/MorphIO) and meshes (via trimesh)
- Write spine tables, morphologies, soma meshes, spine meshes, and spine skeletons
- Validate spine tables against the format specification before writing
- Validate entire morph-with-spines files (structure and optionally data integrity)
- Head/neck triangle classification with filtering (`include_head`, `include_neck`)
- Support for branched spines with multiple heads
- Spine type classification (thin, mushroom, stubby, filopodium, branched, etc.)
- Lazy or eager mesh loading
- Coordinate transformations between local spine and global neuron frames


## Upgrading from v0.x

Version 1.0 drops support for reading spine tables stored as pandas DataFrames (v0.1 format)
inside HDF5 files. If you have files in the old format, convert them before loading:

```bash
python scripts/h5_dataframe_to_h5_datasets_group.py old_file.h5 new_file.h5
```

The conversion script requires the `tables` package:

```bash
pip install morph-spines[scripts]
```


## File format

The morphology-with-spines format is documented in
[`examples/data/README.md`](examples/data/README.md).


## Development

Run tests:

```bash
pytest
```

Lint:

```bash
ruff check src/ tests/
```

Type check:

```bash
mypy src/
```


## Examples

See the [`examples/`](examples/) folder for Jupyter notebooks demonstrating visualization and
usage.


## License

Copyright (c) 2025-2026 Open Brain Institute.

Licensed under Apache-2.0.
