"""Represents a neron morphology with spines.

Provides utility and data acces to a representation of a
neuron morphology with individual spines.
"""

import h5py
import trimesh

from morph_spines.core.h5_schema import *

class Soma:
    """Represents the soma part and its mesh of the morphology with spines format."""
    def __init__(self, meshes_filepath, morphology_name):
        """Default constructor.

        morph_spines.morph_spine_loader.load_spines() intended for users.
        """
        self.name = morphology_name
        self._filepath = meshes_filepath


    @property
    def soma_mesh_points(self):
        """Points of the soma mesh.

        The points (i.e., vertices) of the mesh describing the shape of
        the neuron soma.
        """
        with h5py.File(self._filepath, "r") as h5_file:
            return h5_file[GRP_SOMA][GRP_MESHES][self.name][GRP_VERTICES][:].astype(float)

    @property
    def soma_mesh_triangles(self):
        """Triangles of the soma mesh.

        The triangles (i.e., faces) of the mesh describing the shape of
        the neuron soma.
        """
        with h5py.File(self._filepath, "r") as h5_file:
            return h5_file[GRP_SOMA][GRP_MESHES][self.name][GRP_TRIANGLES][:].astype(int)

    @property
    def soma_mesh(self):
        """Returns the mesh (as a trimesh.Trimesh) of the neuron soma."""
        soma_mesh = trimesh.Trimesh(vertices=self.soma_mesh_points, faces=self.soma_mesh_triangles)
        return soma_mesh
    
    @property
    def center(self):
        return self.soma_mesh_points.mean(axis=0)
