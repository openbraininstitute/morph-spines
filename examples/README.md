# Examples

Jupyter notebooks demonstrating the `morph-spines` library.


## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 01 | `01_getting_started.ipynb` | Core API walkthrough: loading data, spine table, skeletons, meshes, head/neck filtering, spine types, section queries. Uses the small sample dataset. |
| 02 | `02_visualize_spines.ipynb` | Visualize spine meshes for a given section. Uses the full neuron dataset. |
| 03 | `03_visualize_morph_with_spines.ipynb` | Full visualization: neuron morphology mesh combined with spine meshes, section-level coloring. Uses the full neuron dataset. |


## Data

Example data is stored in the `data/` subdirectory:

### `data/morphology_with_spines/`

HDF5 files in the morphology-with-spines format:

- `sample_neurons_with_spines.h5` — Small synthetic sample (not a real neuron) with 2 neurons,
  4 spines each. Includes head/neck triangle classification. Used to illustrate the library API.
  Generated with the `scripts/create_sample_data.py` script.
- `864691134884740346.h5` — Real neuron morphology with spines (larger file, no head/neck
  classification).

### `data/morphology_meshes/`

OBJ mesh files for neuron morphology visualization:

- `sample_neurons_with_spines_neuron_0.obj`, `sample_neurons_with_spines_neuron_1.obj` — Meshes
  for the synthetic neurons.
- `864691134884740346.obj` — Mesh for the real neuron.

### Format documentation

See [`data/README.md`](data/README.md) for the full file format specification.
