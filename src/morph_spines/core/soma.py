"""Represents the soma of a neuron morphology.

Provides utility and data access to the soma mesh of a neuron.
"""

import h5py
import numpy as np
import trimesh
from numpy.typing import NDArray

from morph_spines.core.h5_schema import GRP_MESHES, GRP_SOMA, GRP_TRIANGLES, GRP_VERTICES


class Soma:
    """Represents the soma part and its mesh of the morphology with spines format."""

    def __init__(self, meshes_filepath: str, morphology_name: str) -> None:
        """Default constructor.

        Initializes a new instance of the Soma class with the given parameters.
        """
        self.name = morphology_name
        self._filepath = meshes_filepath
        self._vertices: NDArray | None = None
        self._triangles: NDArray | None = None

    def _load_mesh_data(self) -> None:
        """Load vertices and triangles from the H5 file and cache data."""
        try:
            with h5py.File(self._filepath, "r") as h5_file:
                soma_grp = h5_file[GRP_SOMA][GRP_MESHES][self.name]
                self._vertices = soma_grp[GRP_VERTICES][:].astype(float)
                self._triangles = soma_grp[GRP_TRIANGLES][:].astype(int)
        except KeyError as e:
            raise ValueError(
                f"Soma mesh not found for '{self.name}' in '{self._filepath}'. "
                f"Expected path: /{GRP_SOMA}/{GRP_MESHES}/{self.name}"
            ) from e
        except OSError as e:
            raise ValueError(
                f"Cannot open file '{self._filepath}': {e}"
            ) from e

    @property
    def soma_mesh_points(self) -> NDArray:
        """Points of the soma mesh.

        The points (i.e., vertices) of the mesh describing the shape of
        the neuron soma.
        """
        if self._vertices is None:
            self._load_mesh_data()
        return self._vertices

    @property
    def soma_mesh_triangles(self) -> NDArray:
        """Triangles of the soma mesh.

        The triangles (i.e., faces) of the mesh describing the shape of
        the neuron soma.
        """
        if self._triangles is None:
            self._load_mesh_data()
        return self._triangles

    @property
    def soma_mesh(self) -> trimesh.Trimesh:
        """Returns the mesh (as a trimesh.Trimesh) of the neuron soma."""
        return trimesh.Trimesh(vertices=self.soma_mesh_points, faces=self.soma_mesh_triangles)

    @property
    def center(self) -> NDArray:
        """Returns the center of the soma mesh."""
        return np.mean(self.soma_mesh_points, axis=0)
