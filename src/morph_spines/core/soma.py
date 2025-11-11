"""Represents a neron morphology with spines.

Provides utility and data access to a representation of a
neuron morphology with individual spines.
"""

import h5py
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

    @property
    def soma_mesh_points(self) -> NDArray:
        """Points of the soma mesh.

        The points (i.e., vertices) of the mesh describing the shape of
        the neuron soma.
        """
        with h5py.File(self._filepath, "r") as h5_file:
            return h5_file[GRP_SOMA][GRP_MESHES][self.name][GRP_VERTICES][:].astype(float)

    @property
    def soma_mesh_triangles(self) -> NDArray:
        """Triangles of the soma mesh.

        The triangles (i.e., faces) of the mesh describing the shape of
        the neuron soma.
        """
        with h5py.File(self._filepath, "r") as h5_file:
            return h5_file[GRP_SOMA][GRP_MESHES][self.name][GRP_TRIANGLES][:].astype(int)

    @property
    def soma_mesh(self) -> trimesh.Trimesh:
        """Returns the mesh (as a trimesh.Trimesh) of the neuron soma."""
        soma_mesh = trimesh.Trimesh(vertices=self.soma_mesh_points, faces=self.soma_mesh_triangles)
        return soma_mesh

    @property
    def center(self) -> NDArray:
        """Returns the center of the soma mesh."""
        return self.soma_mesh_points.mean(axis=0)
