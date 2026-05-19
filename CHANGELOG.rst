Changelog
=========

Version 1.0.0 (2026-05-19)
---------------------------

Breaking Changes
~~~~~~~~~~~~~~~~
- Dropped support for reading spine tables stored as pandas DataFrames (v0.1 format).
  Files using the old format must be converted using the
  ``h5_dataframe_to_h5_datasets_group.py`` script before they can be loaded.

New Features
~~~~~~~~~~~~
- Added writer API: ``write_spine_table``, ``write_morphology``, ``write_soma_mesh``,
  ``write_spine_meshes``, ``write_spine_skeletons``.
- Added ``validate_spine_table`` for validating spine table DataFrames against the format spec.
- Added ``load_spine_table`` to the public API.
- Conversion script now writes version metadata to output files.

Improvements
~~~~~~~~~~~~
- Replaced ``print()`` warnings with ``warnings.warn()`` in ``Spines`` class.
- ``Soma`` class now caches mesh data (single file read) and provides clear error messages.
- Added error handling for non-existent or corrupt HDF5 files.
- Moved ``pytest`` and ``pytest-cov`` to optional test dependencies.
- Removed ``tables`` from runtime dependencies (now optional for conversion script).
- Added minimum version pins for all dependencies.
- Fixed typos in module docstrings.


Version 0.7.0 (2026-02-06)
---------------------------

Improvements
~~~~~~~~~~~~
- Improved scripts for creating sample testing data.
- Improved test coverage.


Version 0.6.0 (2026-01-29)
---------------------------

New Features
~~~~~~~~~~~~
- Added option to load all spine meshes at once for better I/O performance.


Version 0.5.0 (2026-01-21)
---------------------------

New Features
~~~~~~~~~~~~
- Extended specification to support spine libraries/collections, allowing the same spine
  skeletons and meshes to be reused across multiple neurites or neurons.


Version 0.4.0 (2025-12-10)
---------------------------

New Features
~~~~~~~~~~~~
- Added support for new spine table format (v1.0): column-wise HDF5 datasets replacing
  pandas DataFrames inside HDF5 files.


Version 0.3.0 (2025-12-10)
---------------------------

New Features
~~~~~~~~~~~~
- Added versioning to the file format.
- First description of the morphology-with-spines file format specification.

Improvements
~~~~~~~~~~~~
- Added unit tests.
- Added type annotations.


Version 0.2.2 (2025-11-03)
---------------------------

Bug Fixes
~~~~~~~~~
- Fixes in Soma class.
- Added function to get neuron position (center) from soma mesh points.


Version 0.2.1 (2025-10-31)
---------------------------

Bug Fixes
~~~~~~~~~
- Added missing dependencies.


Version 0.2.0 (2025-10-31)
---------------------------

New Features
~~~~~~~~~~~~
- Split morphology-with-spines into subcomponents (morphology and spines).


Version 0.1.0 (2025-10-24)
---------------------------

New Features
~~~~~~~~~~~~
- Minimal working version: load morphology with spines from HDF5 files.
