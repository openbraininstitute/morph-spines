# morph-spines

A Python library for loading and accessing neuron morphologies with dendritic spine data from HDF5
files. It provides structured access to spine skeletons, meshes, and spatial transformations.


## Quick example

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


## Installation

```bash
pip install morph-spines
```

For development:

```bash
git clone https://github.com/openbraininstitute/morph-spines.git
cd morph-spines
pip install -e .
```


## Features

- Load neuron morphologies with spine data from HDF5 files
- Access the spine table with per-spine properties (position, orientation, section placement)
- Access spine skeletons (via NeuroM/MorphIO) and meshes (via trimesh)
- Head/neck triangle classification with filtering (`include_head`, `include_neck`)
- Support for branched spines with multiple heads
- Spine type classification (thin, mushroom, stubby, filopodium, branched, etc.)
- Lazy or eager mesh loading
- Coordinate transformations between local spine and global neuron frames


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

Copyright (c) 2025-2026 Open Brain Institute
