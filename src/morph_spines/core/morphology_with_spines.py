"""Represents a neron morphology with spines.

Provides utility and data acces to a representation of a
neuron morphology with individual spines.
"""

import h5py
import trimesh
from scipy.spatial.transform import Rotation

from neurom.core.morphology import Morphology

# Names of groups in the morphology-w-spines hdf5 file
# Top-level groups
GRP_EDGES = "edges"
GRP_MORPH = "morphology"
GRP_SOMA = "soma"
GRP_SPINES = "spines"
# Sub-groups
GRP_MESHES = "meshes"
GRP_SKELETONS = "skeletons"
# Sub-sub-groups inside meshes
GRP_OFFSETS = "offsets"
GRP_TRIANGLES = "triangles"
GRP_VERTICES = "vertices"

# Columns of edge table dataframes
COL_SPINE_MESH = "spine_morphology"
COL_SPINE_ID = "spine_id"
COL_ROTATION = ["spine_rotation_x", "spine_rotation_y", "spine_rotation_z", "spine_rotation_w"]
COL_TRANSLATION = ["afferent_surface_x", "afferent_surface_y", "afferent_surface_z"]
COL_AFF_SEC = "afferent_section_id"

class MorphologyWithSpines(Morphology):
    """Represents spiny neuron morphology.

    A helper class to access the advanced information contained
    in the MorphologyWithSpines format.
    """

    def __init__(
        self,
        meshes_filepath,
        morphology_name,
        morphio_morphology,
        spine_table,
        centered_spine_skeletons,
        spines_are_centered=True,
        process_subtrees=False,
    ):
        """Default constructor.

        morph_spines.morph_spine_loader.load_morphology_with_spines() intended for users.
        """
        super().__init__(
            morphio_morphology, name=morphology_name, process_subtrees=process_subtrees
        )
        self._filepath = meshes_filepath
        self.name = morphology_name
        self._spines_are_centered = spines_are_centered
        self._centered_spine_skeletons = centered_spine_skeletons
        self.spine_table = spine_table

        if self._spines_are_centered:
            self._spine_skeletons = self._transform_spine_skeletons()
        else:
            self._spine_skeletons = self._centered_spine_skeletons

    @property
    def spine_count(self):
        """Number of spines on morphology."""
        return self.spine_table.shape[0]

    def spine_transformations(self, spine_loc):
        """Spine coordinate system transformations.

        Transformations from the local coordinate system of a spine
        (origin near its root, y-axis pointing towards its tip) to the
        global coordinate system of the neuron.
        """
        spine_rotation = Rotation.from_quat(self.spine_table.loc[spine_loc, COL_ROTATION].to_numpy(dtype=float))
        spine_transformation = self.spine_table.loc[spine_loc, COL_TRANSLATION].to_numpy(dtype=float)
        return spine_rotation, spine_transformation

    def transform_for_spine(self, spine_loc, spine_points):
        """Apply spine coordinate system transformations.

        Apply the transformation from the local spine coordinate system
        to the global neuron coordinate system to a set of points.
        """
        spine_rotation, spine_transformation = self.spine_transformations(spine_loc)
        return spine_rotation.apply(spine_points) + spine_transformation.reshape((1, -1))

    def _transform_spine_skeletons(self):
        """Apply transformations to spine skeletons.

        A helper that transforms all centered (in local coordinate system)
        spine skeletons of this class to the global neuron coordinate system.
        """

        spines = self._centered_spine_skeletons.to_morphio().as_mutable()
        assert len(spines.root_sections) == self.spine_table.shape[0]
        for spine_idx, root_spine in enumerate(spines.root_sections):
            lst_in = [root_spine]
            while len(lst_in) > 0:
                lst_out = []
                for section in lst_in:
                    spine_points = self.transform_for_spine(spine_idx, section.points)
                    section.points = spine_points
                    lst_out.extend(section.children)
                lst_in = lst_out
        return Morphology(spines.as_immutable(), name=f"{self.name}_spines")

    @property
    def spine_skeletons(self):
        """The spine skeletons in global coordinates."""
        return self._spine_skeletons.neurites

    @property
    def centered_spine_skeletons(self):
        """The spine skeletons in local coordinates."""
        return self._centered_spine_skeletons.neurites

    def _spine_mesh_points(self, spine_loc, transform=True):
        """Points of spine mesh.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines.
        """
        _spine_mesh_grp = self.spine_table.loc[spine_loc, COL_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[spine_loc, COL_SPINE_ID])
        with h5py.File(self._filepath, "r") as h5_file:
            group = h5_file[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]  # [_spine_id_grp]
            vertex_start = group[GRP_OFFSETS][_spine_id, 0]
            vertex_end = group[GRP_OFFSETS][_spine_id + 1, 0]
            spine_points = group[GRP_VERTICES][vertex_start:vertex_end].astype(float)

        if not transform:
            return spine_points
        return self.transform_for_spine(spine_loc, spine_points)

    def spine_mesh_triangles(self, spine_loc):
        """Triangles of spine mesh.

        The triangles (i.e., faces) of the meshes describing the shape of
        individual spines.
        """
        _spine_mesh_grp = self.spine_table.loc[spine_loc, COL_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[spine_loc, COL_SPINE_ID])
        with h5py.File(self._filepath, "r") as h5_file:
            group = h5_file[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]  # [_spine_id_grp]
            vertex_start = group[GRP_OFFSETS][_spine_id, 1]
            vertex_end = group[GRP_OFFSETS][_spine_id + 1, 1]
            triangles = group[GRP_TRIANGLES][vertex_start:vertex_end].astype(int)
        return triangles

    def spine_mesh_points(self, spine_loc):
        """Points of spine mesh - global.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In global coordinates.
        """
        return self._spine_mesh_points(spine_loc, transform=self._spines_are_centered)

    def centered_mesh_points(self, spine_loc):
        """Points of spine mesh - local.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In local spine coordinates.
        """
        return self._spine_mesh_points(spine_loc, transform=False)

    def spine_mesh(self, spine_loc):
        """Spine mesh representation - global.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In global neuron coordinates.
        """
        spine_mesh = trimesh.Trimesh(
            vertices=self.spine_mesh_points(spine_loc),
            faces=self.spine_mesh_triangles(spine_loc)
        )
        return spine_mesh

    def centered_spine_mesh(self, spine_loc):
        """Spine mesh representation - local.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In local spine coordinates.
        """
        spine_mesh = trimesh.Trimesh(
            vertices=self.centered_mesh_points(spine_loc),
            faces=self.spine_mesh_triangles(spine_loc)
        )
        return spine_mesh

    def spine_indices_for_section(self, section_id):
        """Indices of spines on a given section.

        Returns the indices (indices for .spine_table or .spine_mesh()) of
        spines located on the specified section.
        """
        return self.spine_table_for_section(section_id).index.to_numpy()

    def spine_table_for_section(self, section_id):
        """Table of spines on a given section.

        Returns the rows of the .spine_table for spines located on the
        specified section.
        """
        return self.spine_table.loc[self.spine_table[COL_AFF_SEC] == section_id]

    def spine_meshes_for_section(self, section_id):
        """Spine meshes for a given section

        Iterator that lists the meshes of spines located on the specified
        section.
        """
        for spine_idx in self.spine_indices_for_section(section_id):
            yield self.spine_mesh(spine_idx)

    def compound_spine_mesh_for_section(self, section_id):
        """Single spine mesh for a given section

        A single compound mesh for all spines located on the section is returned.
        """
        return trimesh.util.concatenate(self.spine_meshes_for_section(section_id))

    def centered_spine_meshes_for_section(self, section_id):
        """Centered spine meshes for a given section

        Iterator that lists the meshes of spines located on the specified
        section. Meshes are transformed to be centered and upright.
        """
        for spine_idx in self.spine_indices_for_section(section_id):
            yield self.centered_spine_mesh(spine_idx)

    def compound_centered_spine_mesh_for_section(self, section_id):
        """Single spine mesh for a given section

        A single compound mesh for all spines located on the section is returned.
        Meshes are transformed to be centered and upright.
        """
        return trimesh.util.concatenate(self.centered_spine_meshes_for_section(section_id))

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

class Spines:
    """Represents the spines part and the meshes of the morphology with spines format."""
    def __init__(self, morphology_name, meshes_filepath, spine_table, centered_spine_skeletons,
                 spines_are_centered=True, ):
        """Default constructor.

        morph_spines.morph_spine_loader.load_spines() intended for users.
        """
        self.name = morphology_name
        self._filepath = meshes_filepath
        self.spine_table = spine_table
        self._centered_spine_skeletons = centered_spine_skeletons
        self._spines_are_centered = spines_are_centered

        if self._spines_are_centered:
            self._spine_skeletons = self._transform_spine_skeletons()
        else:
            self._spine_skeletons = self._centered_spine_skeletons

    @property
    def spine_count(self):
        """Number of spines on morphology."""
        return self.spine_table.shape[0]

    def spine_transformations(self, spine_loc):
        """Spine coordinate system transformations.

        Transformations from the local coordinate system of a spine
        (origin near its root, y-axis pointing towards its tip) to the
        global coordinate system of the neuron.
        """
        spine_rotation = Rotation.from_quat(self.spine_table.loc[spine_loc, COL_ROTATION].to_numpy(dtype=float))
        spine_transformation = self.spine_table.loc[spine_loc, COL_TRANSLATION].to_numpy(dtype=float)
        return spine_rotation, spine_transformation

    def transform_for_spine(self, spine_loc, spine_points):
        """Apply spine coordinate system transformations.

        Apply the transformation from the local spine coordinate system
        to the global neuron coordinate system to a set of points.
        """
        spine_rotation, spine_transformation = self.spine_transformations(spine_loc)
        return spine_rotation.apply(spine_points) + spine_transformation.reshape((1, -1))

    def _transform_spine_skeletons(self):
        """Apply transformations to spine skeletons.

        A helper that transforms all centered (in local coordinate system)
        spine skeletons of this class to the global neuron coordinate system.
        """

        spines = self._centered_spine_skeletons.to_morphio().as_mutable()
        assert len(spines.root_sections) == self.spine_table.shape[0]
        for spine_idx, root_spine in enumerate(spines.root_sections):
            lst_in = [root_spine]
            while len(lst_in) > 0:
                lst_out = []
                for section in lst_in:
                    spine_points = self.transform_for_spine(spine_idx, section.points)
                    section.points = spine_points
                    lst_out.extend(section.children)
                lst_in = lst_out
        return Morphology(spines.as_immutable(), name=f"{self.name}_spines")

    @property
    def spine_skeletons(self):
        """The spine skeletons in global coordinates."""
        return self._spine_skeletons.neurites

    @property
    def centered_spine_skeletons(self):
        """The spine skeletons in local coordinates."""
        return self._centered_spine_skeletons.neurites

    def _spine_mesh_points(self, spine_loc, transform=True):
        """Points of spine mesh.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines.
        """
        _spine_mesh_grp = self.spine_table.loc[spine_loc, COL_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[spine_loc, COL_SPINE_ID])
        with h5py.File(self._filepath, "r") as h5_file:
            group = h5_file[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]  # [_spine_id_grp]
            vertex_start = group[GRP_OFFSETS][_spine_id, 0]
            vertex_end = group[GRP_OFFSETS][_spine_id + 1, 0]
            spine_points = group[GRP_VERTICES][vertex_start:vertex_end].astype(float)

        if not transform:
            return spine_points
        return self.transform_for_spine(spine_loc, spine_points)

    def spine_mesh_triangles(self, spine_loc):
        """Triangles of spine mesh.

        The triangles (i.e., faces) of the meshes describing the shape of
        individual spines.
        """
        _spine_mesh_grp = self.spine_table.loc[spine_loc, COL_SPINE_MESH]
        _spine_id = int(self.spine_table.loc[spine_loc, COL_SPINE_ID])
        with h5py.File(self._filepath, "r") as h5_file:
            group = h5_file[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]  # [_spine_id_grp]
            vertex_start = group[GRP_OFFSETS][_spine_id, 1]
            vertex_end = group[GRP_OFFSETS][_spine_id + 1, 1]
            triangles = group[GRP_TRIANGLES][vertex_start:vertex_end].astype(int)
        return triangles

    def spine_mesh_points(self, spine_loc):
        """Points of spine mesh - global.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In global coordinates.
        """
        return self._spine_mesh_points(spine_loc, transform=self._spines_are_centered)

    def centered_mesh_points(self, spine_loc):
        """Points of spine mesh - local.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In local spine coordinates.
        """
        return self._spine_mesh_points(spine_loc, transform=False)

    def spine_mesh(self, spine_loc):
        """Spine mesh representation - global.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In global neuron coordinates.
        """
        spine_mesh = trimesh.Trimesh(
            vertices=self.spine_mesh_points(spine_loc),
            faces=self.spine_mesh_triangles(spine_loc)
        )
        return spine_mesh

    def centered_spine_mesh(self, spine_loc):
        """Spine mesh representation - local.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In local spine coordinates.
        """
        spine_mesh = trimesh.Trimesh(
            vertices=self.centered_mesh_points(spine_loc),
            faces=self.spine_mesh_triangles(spine_loc)
        )
        return spine_mesh

    def spine_indices_for_section(self, section_id):
        """Indices of spines on a given section.

        Returns the indices (indices for .spine_table or .spine_mesh()) of
        spines located on the specified section.
        """
        return self.spine_table_for_section(section_id).index.to_numpy()

    def spine_table_for_section(self, section_id):
        """Table of spines on a given section.

        Returns the rows of the .spine_table for spines located on the
        specified section.
        """
        return self.spine_table.loc[self.spine_table[COL_AFF_SEC] == section_id]

    def spine_meshes_for_section(self, section_id):
        """Spine meshes for a given section

        Iterator that lists the meshes of spines located on the specified
        section.
        """
        for spine_idx in self.spine_indices_for_section(section_id):
            yield self.spine_mesh(spine_idx)

    def compound_spine_mesh_for_section(self, section_id):
        """Single spine mesh for a given section

        A single compound mesh for all spines located on the section is returned.
        """
        return trimesh.util.concatenate(self.spine_meshes_for_section(section_id))

    def centered_spine_meshes_for_section(self, section_id):
        """Centered spine meshes for a given section

        Iterator that lists the meshes of spines located on the specified
        section. Meshes are transformed to be centered and upright.
        """
        for spine_idx in self.spine_indices_for_section(section_id):
            yield self.centered_spine_mesh(spine_idx)

    def compound_centered_spine_mesh_for_section(self, section_id):
        """Single spine mesh for a given section

        A single compound mesh for all spines located on the section is returned.
        Meshes are transformed to be centered and upright.
        """
        return trimesh.util.concatenate(self.centered_spine_meshes_for_section(section_id))

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
