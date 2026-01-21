"""Represents the spines of a neuron morphology with spines.

Provides utility and data access to a representation of a
neuron morphology with individual spines.
"""

from collections.abc import Iterator

import h5py
import numpy
import pandas
import trimesh
from neurom.core.morphology import Morphology, Neurite
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from morph_spines.core.h5_schema import (
    COL_AFF_SEC,
    COL_SPINE_ID,
    COL_SPINE_MORPH,
    GRP_MESHES,
    GRP_OFFSETS,
    GRP_SPINES,
    GRP_TRIANGLES,
    GRP_VERTICES,
)
from morph_spines.utils import geometry


class Spines:
    """Represents the spines part and the meshes of the morphology with spines format."""

    def __init__(
        self,
        meshes_filepath: str,
        morphology_name: str,
        spine_table: pandas.DataFrame,
        centered_spine_skeletons: Morphology,
        spines_are_centered: bool = True,
        spine_meshes: list[trimesh.Trimesh] | None = None,
    ) -> None:
        """Default constructor.

        Initializes a new instance of the Spines class with the given parameters.
        """
        self._filepath = meshes_filepath
        self.name = morphology_name
        self.spine_table = spine_table
        self._centered_spine_skeletons = centered_spine_skeletons
        self._spines_are_centered = spines_are_centered
        self._spine_meshes = spine_meshes if spine_meshes is not None else []

        if self._spines_are_centered:
            self._spine_skeletons = self._transform_spine_skeletons()
        else:
            self._spine_skeletons = self._centered_spine_skeletons

    @property
    def spine_count(self) -> int:
        """Number of spines on morphology."""
        return self.spine_table.shape[0]

    def spine_transformations(self, spine_loc: int) -> tuple[Rotation, NDArray]:
        """Spine coordinate system transformations.

        Transformations from the local coordinate system of a spine
        (origin near its root, y-axis pointing towards its tip) to the
        global coordinate system of the neuron.
        """
        return geometry.spine_transformations(self.spine_table, spine_loc)

    def transform_for_spine(self, spine_loc: int, spine_points: NDArray) -> NDArray:
        """Apply spine coordinate system transformations.

        Apply the transformation from the local spine coordinate system
        to the global neuron coordinate system to a set of points.
        """
        return geometry.transform_for_spine(self.spine_table, spine_loc, spine_points)

    def _transform_spine_skeletons(self) -> Morphology:
        """Apply transformations to spine skeletons.

        A helper that transforms all centered (in local coordinate system)
        spine skeletons of this class to the global neuron coordinate system.
        """
        spines = self._centered_spine_skeletons.to_morphio().as_mutable()
        if len(spines.root_sections) != self.spine_table.shape[0]:
            raise ValueError(
                f"Number of root sections ({len(spines.root_sections)}) "
                f"does not match spine table rows ({self.spine_table.shape[0]})."
            )

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
    def spine_skeletons(self) -> Iterator[Neurite]:
        """The spine skeletons in global coordinates."""
        return self._spine_skeletons.neurites

    @property
    def centered_spine_skeletons(self) -> Iterator[Neurite]:
        """The spine skeletons in local coordinates."""
        return self._centered_spine_skeletons.neurites

    def _spine_mesh_points(self, spine_loc: int, transform: bool = True) -> NDArray:
        """Points of spine mesh.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines.
        """
        _spine_row = self.spine_table.loc[spine_loc]
        _spine_mesh_grp = _spine_row[COL_SPINE_MORPH]
        _spine_id = int(_spine_row[COL_SPINE_ID])

        if len(self._spine_meshes) != 0:
            spine_points = numpy.array(self._spine_meshes[_spine_id].vertices)

            if not transform:
                # Spine mesh points are already in global coordinates, so we need to convert them
                # back to the local spine coordinate system
                spine_points = geometry.inverse_transform_for_spine(
                    self.spine_table, spine_loc, spine_points
                )

        else:
            with h5py.File(self._filepath, "r") as h5_file:
                group = h5_file[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]
                vertex_start = group[GRP_OFFSETS][_spine_id, 0]
                vertex_end = group[GRP_OFFSETS][_spine_id + 1, 0]
                spine_points = group[GRP_VERTICES][vertex_start:vertex_end].astype(float)

            if transform:
                spine_points = geometry.transform_for_spine(
                    self.spine_table, spine_loc, spine_points
                )

        return spine_points

    def spine_mesh_triangles(self, spine_loc: int) -> NDArray:
        """Triangles of spine mesh.

        The triangles (i.e., faces) of the meshes describing the shape of
        individual spines.
        """
        _spine_row = self.spine_table.loc[spine_loc]
        _spine_mesh_grp = _spine_row[COL_SPINE_MORPH]
        _spine_id = int(_spine_row[COL_SPINE_ID])

        if len(self._spine_meshes) != 0:
            triangles = self._spine_meshes[_spine_id].triangles

        else:
            with h5py.File(self._filepath, "r") as h5_file:
                group = h5_file[GRP_SPINES][GRP_MESHES][_spine_mesh_grp]
                triangle_start = group[GRP_OFFSETS][_spine_id, 1]
                triangle_end = group[GRP_OFFSETS][_spine_id + 1, 1]
                triangles = group[GRP_TRIANGLES][triangle_start:triangle_end].astype(int)

        return triangles

    def spine_mesh_points(self, spine_loc: int) -> NDArray:
        """Points of spine mesh - global.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In global coordinates.
        """
        return self._spine_mesh_points(spine_loc, transform=self._spines_are_centered)

    def centered_mesh_points(self, spine_loc: int) -> NDArray:
        """Points of spine mesh - local.

        The points (i.e., vertices) of the meshes describing the shape of
        individual spines. In local spine coordinates.
        """
        return self._spine_mesh_points(spine_loc, transform=False)

    def spine_mesh(self, spine_loc: int) -> trimesh.Trimesh:
        """Spine mesh representation - global.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In global neuron coordinates.
        """
        if len(self._spine_meshes) != 0:
            spine_mesh = self._spine_meshes[spine_loc]
        else:
            spine_mesh = trimesh.Trimesh(
                vertices=self.spine_mesh_points(spine_loc),
                faces=self.spine_mesh_triangles(spine_loc),
            )
        return spine_mesh

    def centered_spine_mesh(self, spine_loc: int) -> trimesh.Trimesh:
        """Spine mesh representation - local.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In local spine coordinates.
        """
        spine_mesh = trimesh.Trimesh(
            vertices=self.centered_mesh_points(spine_loc),
            faces=self.spine_mesh_triangles(spine_loc),
        )
        return spine_mesh

    def spine_indices_for_section(self, section_id: int) -> NDArray:
        """Indices of spines on a given section.

        Returns the indices (indices for .spine_table or .spine_mesh()) of
        spines located on the specified section.
        """
        return self.spine_table_for_section(section_id).index.to_numpy()

    def spine_table_for_section(self, section_id: int) -> pandas.DataFrame:
        """Table of spines on a given section.

        Returns the rows of the .spine_table for spines located on the
        specified section.
        """
        return self.spine_table.loc[self.spine_table[COL_AFF_SEC] == section_id]

    def spine_meshes_for_section(self, section_id: int) -> Iterator[trimesh.Trimesh]:
        """Spine meshes for a given section.

        Iterator that lists the meshes of spines located on the specified
        section.
        """
        for spine_idx in self.spine_indices_for_section(section_id):
            yield self.spine_mesh(spine_idx)

    def compound_spine_mesh_for_section(self, section_id: int) -> trimesh.Trimesh:
        """Single spine mesh for a given section.

        A single compound mesh for all spines located on the section is returned.
        """
        return trimesh.util.concatenate(self.spine_meshes_for_section(section_id))

    def centered_spine_meshes_for_section(self, section_id: int) -> Iterator[trimesh.Trimesh]:
        """Centered spine meshes for a given section.

        Iterator that lists the meshes of spines located on the specified
        section. Meshes are transformed to be centered and upright.
        """
        for spine_idx in self.spine_indices_for_section(section_id):
            yield self.centered_spine_mesh(spine_idx)

    def compound_centered_spine_mesh_for_section(self, section_id: int) -> trimesh.Trimesh:
        """Single spine mesh for a given section.

        A single compound mesh for all spines located on the section is returned.
        Meshes are transformed to be centered and upright.
        """
        return trimesh.util.concatenate(self.centered_spine_meshes_for_section(section_id))

    def spine_meshes_for_morphology(self) -> Iterator[trimesh.Trimesh]:
        """Return all the spine meshes of the morphology.

        An array of spine meshes is returned. The meshes are already rotated and translated with
        respect to the global morphology coordinates. There is an implicit index for each spine
        mesh that matches the spine index order from the spine table.
        """
        yield from self._spine_meshes

    def compound_spine_meshes_for_morphology(self) -> trimesh.Trimesh:
        """Return all the spine meshes of the morphology unified into a single mesh.

        The mesh is already rotated and translated with respect to the global morphology
        coordinates.
        """
        return trimesh.util.concatenate(self.spine_meshes_for_morphology())

    def centered_spine_meshes_for_morphology(self) -> Iterator[trimesh.Trimesh]:
        """Return all the spine meshes of the morphology.

        An array of spine meshes is returned. The meshes are in local spine coordinates.
        There is an implicit index for each spine mesh that matches the spine index order
        from the spine table.
        """
        for spine_idx, spine_mesh in enumerate(self._spine_meshes):
            centered_spine_mesh = trimesh.Trimesh(
                vertices=geometry.inverse_transform_for_spine(
                    self.spine_table, spine_idx, spine_mesh.vertices
                ),
                faces=spine_mesh.triangles,
            )
            yield centered_spine_mesh

    def compound_centered_spine_meshes_for_morphology(self) -> trimesh.Trimesh:
        """Return all the spine meshes of the morphology unified into a single mesh.

        The mesh is already rotated and translated with respect to the global morphology
        coordinates.
        """
        return trimesh.util.concatenate(self.centered_spine_meshes_for_morphology())
