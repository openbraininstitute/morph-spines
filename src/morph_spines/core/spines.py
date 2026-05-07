"""Represents the spines of a neuron morphology with spines.

Provides utility and data access to a representation of a
neuron morphology with individual spines.
"""

from collections.abc import Iterator

import h5py
import numpy as np
import pandas
import trimesh
from neurom.core.morphology import Morphology, Neurite
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from morph_spines.core.h5_schema import (
    COL_AFF_SEC,
    COL_ROTATION,
    COL_SPINE_ID,
    COL_SPINE_MORPH,
    COL_SPINE_TYPE,
    COL_TRANSLATION,
    GRP_HEAD_NECK_VALUES,
    GRP_MESHES,
    GRP_OFFSETS,
    GRP_SPINES,
    GRP_TRIANGLES,
    GRP_VERTICES,
    OFF_COL_HEAD_NECK,
    OFF_COL_TRIANGLES,
    OFF_COL_VERTICES,
)
from morph_spines.core.spine_type import SpineType
from morph_spines.utils import geometry, mesh


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
        head_neck_offsets: list[NDArray] | None = None,
    ) -> None:
        """Default constructor.

        Initializes a new instance of the Spines class with the given parameters.
        """
        self._filepath = meshes_filepath
        self.name = morphology_name
        self.spine_table = spine_table
        self._centered_spine_skeletons = centered_spine_skeletons
        self._spines_are_centered = spines_are_centered

        self._valid_head_neck_offsets = True

        if spine_meshes is not None and head_neck_offsets is None:
            print(
                "WARNING: no head_neck_offsets provided, spine head/neck classification won't be "
                "available"
            )
            self._valid_head_neck_offsets = False

        self._spine_meshes = spine_meshes if spine_meshes is not None else []
        self._head_neck_offsets = head_neck_offsets if head_neck_offsets is not None else []

        if head_neck_offsets is not None and spine_meshes is None:
            print(
                "WARNING: head_neck_offsets value will be ignored because spine_meshes were not"
                " provided",
            )
            self._head_neck_offsets = []

        self._head_neck_offsets_checked = len(self._head_neck_offsets) > 0

        if self._spines_are_centered:
            self._spine_skeletons = self._transform_spine_skeletons()
        else:
            self._spine_skeletons = self._centered_spine_skeletons

    @property
    def spine_count(self) -> int:
        """Number of spines on morphology."""
        return len(self.spine_table)

    def spine_type(self, spine_loc: int) -> SpineType:
        """Morphological type of a spine.

        Returns the spine type classification for the given spine. If the
        spine table does not contain a spine_type column, returns
        SpineType.UNDEFINED.
        """
        if COL_SPINE_TYPE not in self.spine_table.columns:
            return SpineType.UNDEFINED
        return SpineType(self.spine_table.loc[spine_loc, COL_SPINE_TYPE])

    def spine_transformations(self, spine_loc: int) -> tuple[Rotation, NDArray]:
        """Spine coordinate system transformations.

        Transformations from the local coordinate system of a spine
        (origin near its root, y-axis pointing towards its tip) to the
        global coordinate system of the neuron.
        """
        spine_row = self.spine_table.loc[spine_loc]
        spine_rotation = Rotation.from_quat(np.array(spine_row[COL_ROTATION].to_numpy(dtype=float)))
        spine_translation = spine_row[COL_TRANSLATION].to_numpy(dtype=float)

        return spine_rotation, spine_translation

    def transform_for_spine(self, spine_loc: int, spine_points: NDArray) -> NDArray:
        """Apply spine coordinate system transformations.

        Apply the transformation from the local spine coordinate system
        to the global neuron coordinate system to a set of points.
        """
        spine_rotation, spine_translation = self.spine_transformations(spine_loc)
        return geometry.transform_for_spine(spine_rotation, spine_translation, spine_points)

    def _transform_spine_skeletons(self) -> Morphology:
        """Apply transformations to spine skeletons.

        A helper that transforms all centered (in local coordinate system)
        spine skeletons of this class to the global neuron coordinate system.
        """
        spines = self._centered_spine_skeletons.to_morphio().as_mutable()
        if len(spines.root_sections) != self.spine_count:
            raise ValueError(
                f"Number of root sections ({len(spines.root_sections)}) "
                f"does not match spine table rows ({self.spine_count})."
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

        The points (i.e., vertices) of the meshes describing the shape of individual spines in
        local (transform=False) or global (transform=True) coordinates.
        """
        if len(self._spine_meshes) != 0:
            spine_points = np.array(self._spine_meshes[spine_loc].vertices)

            if not transform:
                # Spine mesh points are already in global coordinates, so we need to convert them
                # back to the local spine coordinate system
                spine_rotation, spine_translation = self.spine_transformations(spine_loc)
                spine_points = geometry.inverse_transform_for_spine(
                    spine_rotation, spine_translation, spine_points
                )

        else:
            spine_row = self.spine_table.loc[spine_loc]
            spine_mesh_grp = spine_row[COL_SPINE_MORPH]
            spine_idx = int(spine_row[COL_SPINE_ID])
            with h5py.File(self._filepath, "r") as h5_file:
                group = h5_file[GRP_SPINES][GRP_MESHES][spine_mesh_grp]
                vertex_start, vertex_end = group[GRP_OFFSETS][
                    spine_idx : spine_idx + 2, OFF_COL_VERTICES
                ]
                spine_points = group[GRP_VERTICES][vertex_start:vertex_end].astype(float)

            if transform:
                spine_rotation, spine_translation = self.spine_transformations(spine_loc)
                spine_points = geometry.transform_for_spine(
                    spine_rotation, spine_translation, spine_points
                )

        return spine_points

    def spine_mesh_triangles(self, spine_loc: int) -> NDArray:
        """Triangles of spine mesh.

        The triangles (i.e., faces) of the meshes describing the shape of
        individual spines.
        """
        if len(self._spine_meshes) == 0:
            spine_row = self.spine_table.loc[spine_loc]
            spine_mesh_grp = spine_row[COL_SPINE_MORPH]
            spine_idx = int(spine_row[COL_SPINE_ID])
            with h5py.File(self._filepath, "r") as h5_file:
                group = h5_file[GRP_SPINES][GRP_MESHES][spine_mesh_grp]
                triangle_start, triangle_end = group[GRP_OFFSETS][
                    spine_idx : spine_idx + 2, OFF_COL_TRIANGLES
                ]
                triangles = group[GRP_TRIANGLES][triangle_start:triangle_end].astype(int)
        else:
            triangles = self._spine_meshes[spine_loc].faces

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

    def _get_head_neck_offsets(self, spine_loc: int) -> NDArray:
        """Get the head/neck triangle offsets for a spine.

        Returns:
        - If no head/neck information is found:
          Returns an empty array

        - Otherwise:
          Returns an offset-style array of length H + 2 for a spine with H heads:
          - Undefined triangles: triangles[0 : offsets[0]]
          - Neck triangles: triangles[offsets[0] : offsets[1]]
          - N-th head: triangles[offsets[N+1] : offsets[N+2]]
          - total number of spine triangles: offsets[-1]

        The head/neck data is stored using a double-index approach: a flat
        head_neck_values dataset holds all offsets concatenated, and
        column 2 of the offsets dataset indexes into it per spine.
        """
        # Use cached offsets if available
        if len(self._head_neck_offsets) != 0:
            return self._head_neck_offsets[spine_loc]

        # Already checked the file and found no offsets
        if not self._valid_head_neck_offsets:
            return np.array([], dtype=int)

        spine_row = self.spine_table.loc[spine_loc]
        spine_mesh_grp = spine_row[COL_SPINE_MORPH]
        spine_idx = int(spine_row[COL_SPINE_ID])

        with h5py.File(self._filepath, "r") as h5_file:
            group = h5_file[GRP_SPINES][GRP_MESHES][spine_mesh_grp]
            offsets_ds = group[GRP_OFFSETS]

            # Old files with only 2 columns (vertices, triangles) have no head/neck classification
            if GRP_HEAD_NECK_VALUES not in group or offsets_ds.shape[1] < 3:
                self._valid_head_neck_offsets = False
                return np.array([], dtype=int)

            hn_start, hn_end = offsets_ds[spine_idx : spine_idx + 2, OFF_COL_HEAD_NECK]
            # Head/neck classification present in the file, but undefined for this spine
            if hn_start == hn_end:
                return np.array([], dtype=int)

            return np.array(group[GRP_HEAD_NECK_VALUES][hn_start:hn_end], dtype=int)

    def spine_mesh(
        self,
        spine_loc: int,
        *,
        include_head: bool = True,
        include_neck: bool = True,
    ) -> trimesh.Trimesh:
        """Spine mesh representation - global.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In global neuron coordinates.

        Args:
            spine_loc: Spine index in the spine table.
            include_head: If False, head triangles are excluded. Default True.
            include_neck: If False, neck triangles are excluded. Default True.

        Returns: trimesh.Trimesh of the spine mesh.

        Raises:
            ValueError: If both include_head and include_neck are False.
        """
        if not include_head and not include_neck:
            raise ValueError("At least one of include_head or include_neck must be True")

        # When both flags are True and meshes are preloaded, return directly
        if include_head and include_neck and len(self._spine_meshes) != 0:
            return self._spine_meshes[spine_loc]

        vertices = self.spine_mesh_points(spine_loc)
        triangles = self.spine_mesh_triangles(spine_loc)

        # We load & cache the whole spine mesh, needs splitting if head or neck need to be skipped
        if not include_head or not include_neck:
            head_neck_offsets = self._get_head_neck_offsets(spine_loc)
            filtered = mesh.filter_triangles_by_head_neck(
                triangles, head_neck_offsets, include_head, include_neck
            )
            triangles = np.concatenate(filtered) if filtered else np.empty((0, 3), dtype=int)
            vertices, triangles = mesh.submesh(vertices, triangles)

        return trimesh.Trimesh(vertices=vertices, faces=triangles)

    def centered_spine_mesh(
        self,
        spine_loc: int,
        *,
        include_head: bool = True,
        include_neck: bool = True,
    ) -> trimesh.Trimesh:
        """Spine mesh representation - local.

        Returns the mesh (as a trimesh.Trimesh) of an individual spine.
        In local spine coordinates.

        Args:
            spine_loc: Spine index in the spine table.
            include_head: If False, head triangles are excluded. Default True.
            include_neck: If False, neck triangles are excluded. Default True.

        Returns: trimesh.Trimesh of the spine mesh in local coordinates.

        Raises:
            ValueError: If both include_head and include_neck are False.
        """
        if not include_head and not include_neck:
            raise ValueError("At least one of include_head or include_neck must be True")

        # Even if meshes are loaded, they're in global coordinates, so we need to transform the
        # points into local coordinates before creating a new mesh.
        # However, there's an exception to this case: when the initial H5 data is not centered, in
        # which case centered data equals the non-centered H5 data.
        if not self._spines_are_centered:
            centered_spine_mesh = self.spine_mesh(
                spine_loc, include_head=include_head, include_neck=include_neck
            )
        else:
            centered_spine_mesh = self.spine_mesh(
                spine_loc, include_head=include_head, include_neck=include_neck
            ).copy()
            spine_rotation, spine_translation = self.spine_transformations(spine_loc)
            transform_matrix = geometry.inverse_transform_matrix_for_spine(
                spine_rotation, spine_translation
            )
            centered_spine_mesh.apply_transform(transform_matrix)

        return centered_spine_mesh

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

    def _load_meshes(self) -> None:
        """Load spine meshes and cache them in memory."""
        spine_meshes = []
        head_neck_offsets = []

        for spine_loc in range(len(self.spine_table)):
            spine_meshes.append(self.spine_mesh(spine_loc))
            head_neck_offsets.append(self._get_head_neck_offsets(spine_loc))

        self._spine_meshes = spine_meshes
        self._head_neck_offsets = head_neck_offsets

    def spine_meshes_for_morphology(self) -> Iterator[trimesh.Trimesh]:
        """Return all the spine meshes of the morphology.

        An array of spine meshes is returned. The meshes are already rotated and translated with
        respect to the global morphology coordinates. There is an implicit index for each spine
        mesh that matches the spine index order from the spine table.
        """
        # If meshes are not loaded, load them now
        if len(self._spine_meshes) == 0:
            self._load_meshes()

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
        # If meshes are not loaded, load them now
        if len(self._spine_meshes) == 0:
            self._load_meshes()

        for spine_idx in range(self.spine_count):
            yield self.centered_spine_mesh(spine_idx)
