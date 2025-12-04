"""This script copies an existing morphology-with-spines file into a new file.

It replaces the existing /edges table (in pandas dataframe format) with an HDF5 group that contains
as many datasets as columns. The dataframe data is stored column-wise in these datasets. The rest
of the groups and datasets are copied untouched.

Usage:
    python h5_dataframe_to_h5_datasets_group.py <input_filepath> <output_filepath>
"""

import sys
from collections.abc import Iterator

import h5py
import pandas as pd
from numpy.typing import NDArray

from morph_spines.utils import morph_spine_loader


def find_datasets(group: h5py.Group) -> Iterator[str]:
    """Recursively find datasets from a given HDF5 group.

    Args:
        group: An HDF5 group.

    Returns: An HDF5 dataset.

    """
    """Recursively yield dataset paths under an HDF5 group."""
    for _key, item in group.items():
        if isinstance(item, h5py.Dataset):
            yield item.name
        elif isinstance(item, h5py.Group):
            yield from find_datasets(item)


def is_pandas_dataframe_group(group: h5py.Group) -> bool:
    """Check if the given HDF5 group contains a pandas DataFrame at its root.

    Args:
        group: An HDF5 group.

    Returns: True if the given HDF5 group contains a pandas DataFrame at its root, False otherwise.

    """
    if isinstance(group, h5py.Group):
        if "pandas_type" in group.attrs:
            return True
    return False


def find_pandas_dataframe_groups(group: h5py.Group) -> Iterator[str]:
    """Recursively yield group paths under an HDF5 group that contain a pandas DataFrame.

    Args:
        group: An HDF5 group.

    Returns: A group containing a pandas DataFrame.

    """
    for _key, item in group.items():
        if isinstance(item, h5py.Group):
            if "pandas_type" in item.attrs:
                yield item.name
            else:
                yield from find_pandas_dataframe_groups(item)


def scan_group_contents(
    group: h5py.Group, pandas_groups: list[str], to_copy: list[str]
) -> tuple[list[str], list[str]]:
    """Recursively scan an HDF5 group.

    Args:
        group: The HDF5 group to scan.
        pandas_groups: List of pandas groups already found, new groups are appended to this list.
        to_copy: List of HDF5 group paths to copy without modification.

    Returns: A list of pandas groups

    """
    for _key, item in group.items():
        if isinstance(item, h5py.Group):
            if is_pandas_dataframe_group(item):
                pandas_groups.append(item.name)
                # Stop recursivity: all groups/datasets under this path represent the dataframe
            elif len(item) == 0:
                # Empty group, we still need to copy it, but we don't continue recursivity
                to_copy.append(item)
            else:
                # Recursively check contents under the group
                pandas_groups, to_copy = scan_group_contents(item, pandas_groups, to_copy)
        elif isinstance(item, h5py.Dataset):
            # Add the dataset to the list of items to copy
            to_copy.append(item.name)

    return pandas_groups, to_copy


def dataframe_to_array_list(df: pd.DataFrame) -> dict[str, NDArray]:
    """Convert the given pandas DataFrame to a list of numpy arrays.

    Args:
        df: The dataframe to convert to list of numpy arrays.

    Returns: A dictionary with as many elements as dataframe columns, where:
    - Key: a dataframe column name
    - Value: the contents of that column in the dataframe

    """
    columns = dict()

    for col in df.columns:
        columns[col] = df[col].to_numpy()

    return columns


def array_list_to_dataframe(filename: str, array_grp_path: str) -> pd.DataFrame:
    """Convert a group of datasets (from HDF5 file) into a pandas DataFrame.

    Args:
        filename: HDF5 filepath.
        array_grp_path: Path where the set of datasets are stored.

    Returns: a pandas DataFrame containing the information of the HDF5 group datasets.

    """
    return morph_spine_loader.load_spine_table(filename, array_grp_path)


def convert_file(input_file: str, output_file: str) -> list[str]:
    """Copy an HDF5 file into a new one, replacing all pandas dataframes with a group of datasets.

    Optimization: dataframes are only searched inside /edges top-level group.

    Args:
        input_file: Input HDF5 file, with pandas dataframes stored in it.
        output_file: New HDF5 file, dataframes replaced with group datasets (under the same path).

    Returns: The list of HDF5 paths in which pandas dataframes were found and converted.

    """
    pandas_groups: list[str] = []

    with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:
        for name, _item in f_in.items():
            if name != "edges":
                # Copy other top-level groups/datasets
                f_in.copy(name, f_out)
            else:
                edges_in = f_in["edges"]
                edges_out = f_out.create_group("edges")

                # At the moment, /edges has only groups with pandas dataframes, but let's be
                # generic, just in case there would be other groups: we need to copy all the groups
                # that don't contain dataframes + convert the dataframes into group datasets
                to_copy: list[str] = []
                pandas_groups, to_copy = scan_group_contents(edges_in, pandas_groups, to_copy)

                # Copy all the groups/datasets that do not represent DataFrames as they are
                for path in to_copy:
                    edges_in.copy(path, edges_out)

                # Replace the pandas dataframes with converted group datasets in the output file
                # (under the same path)
                for df_path in pandas_groups:
                    df_in = pd.read_hdf(input_file, key=df_path)
                    df_in = df_in.to_frame() if isinstance(df_in, pd.Series) else df_in
                    dict_out = dataframe_to_array_list(df_in)
                    dset_group = edges_out.create_group(df_path)
                    for col_name, col_data in dict_out.items():
                        dset_name = str(f"{dset_group.name}/{col_name}")
                        dset_group.create_dataset(dset_name, data=col_data)

    if len(pandas_groups) == 0:
        print(f"No pandas dataframes found, file copied untouched: {output_file}")
    else:
        print(f"Converted the following pandas dataframes: {pandas_groups}")

    return pandas_groups


def test_equal_dataframes(in_file: str, out_file: str, df_path: str) -> bool:
    """Check that 2 dataframes encoded differently in H5 files are equal.

    Note: the column order is ignored, as HDF5 Iterators return objects alphabetically ordered.

    Args:
        in_file: First file with data stored as pandas dataframe.
        out_file: Second file with data stored as group datasets.
        df_path: Path within the files where the two structures are located.

    Returns: True if the structured array converted to a dataframe is equal to the first dataframe,
    False otherwise.

    """
    in_spine_table = morph_spine_loader.load_spine_table(in_file, df_path).sort_index(axis=1)
    out_spine_table = array_list_to_dataframe(out_file, df_path).sort_index(axis=1)

    if in_spine_table.equals(out_spine_table):
        return True
    else:
        return False


def validate_conversion(input_file: str, output_file: str, pandas_groups: list[str]) -> None:
    """Validate that the converted dataframes into structured arrays match the original data.

    Args:
        input_file: Input HDF5 file, with pandas dataframes stored in it.
        output_file: New HDF5 file, pandas dataframes replaced with structured arrays.
        pandas_groups: List of pandas dataframes that were converted.

    Returns: None

    """
    if len(pandas_groups) > 0:
        errors = [
            (p_df, test_equal_dataframes(input_file, output_file, p_df)) for p_df in pandas_groups
        ]
        if all(result for (_, result) in errors):
            print(f"File converted successfully, all dataframe conversions match: {output_file}")
        else:
            raise RuntimeError(
                f"An error occurred, the original dataframes do not match the "
                f"converted ones in the new {output_file}: {errors}"
            )


def main() -> None:
    """Main function.

    Returns: None

    """
    if len(sys.argv) != 3:
        raise RuntimeError(
            f"Usage: python h5_dataframe_to_h5_struct_array.py <input_filepath> <output_filepath>, "
            f"but got {len(sys.argv)} args"
        )

    (_, input_file, output_file) = sys.argv

    pandas_groups = convert_file(input_file, output_file)
    validate_conversion(input_file, output_file, pandas_groups)


if __name__ == "__main__":
    main()
