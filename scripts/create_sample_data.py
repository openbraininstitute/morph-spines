"""This script creates a morphology-with-spines file with fake sample data.

See: <python create_sample_data.py -h> for help on usage.
"""

import argparse

import create_sample_data_writer as writer
import create_sample_morphology_data as morph_creator
import create_sample_spines_data as spine_creator


def create_sample_data(
    output_file: str,
    num_neurons: int,
    num_colls: int,
    num_spines: int,
    group_by_neuron: bool,
    centered_spines: bool,
) -> None:
    """Generate the sample data and write it to the given output file.

    Creates as many neurons as given by parameter and as many spines per neuron or per collection.
    The spines can be grouped by neuron (group_by_neuron=True) or by collection
    (group_by_neuron=False). The parameter num_colls has no effect if spines are grouped by neuron.
    The spines can be centered at the origin or at its original position.

    Args:
        output_file: Filepath to output file that will be created
        num_neurons: Number of neurons to be created
        num_colls: Number of spine collections to be created, if spines grouped by collection
        num_spines: Number of spines per collection or per neuron
        group_by_neuron: Whether to group spines by neuron or by collection
        centered_spines: Whether to place spines centered at origin or at its original position

    Returns: None
    """
    neuron_names = [f"neuron_{i}" for i in range(num_neurons)]
    collection_names = (
        neuron_names if group_by_neuron else [f"collection_{i}" for i in range(num_colls)]
    )

    morph_skeletons = morph_creator.generate_neuron_skeletons(neuron_names)
    soma_meshes = morph_creator.generate_soma_meshes_arrays(morph_skeletons)
    spines_skeletons = spine_creator.generate_all_spines_skeletons(collection_names, num_spines)
    spines_tables = spine_creator.generate_spines_tables(
        morph_skeletons, spines_skeletons, group_by_neuron
    )

    morph_meshes = morph_creator.generate_neuron_meshes(morph_skeletons)
    # FIXME: to be implemented!!
    # if not centered_spines:
    #    spines_skeletons = spine_creator.transform_spines_skeletons(
    #        spines_tables, spines_skeletons
    #    )

    spines_meshes = spine_creator.generate_all_spines_meshes(spines_skeletons)

    data = {
        "spine_tables": spines_tables,
        "neuron_skeletons": morph_skeletons,
        "soma_meshes": soma_meshes,
        "spines_meshes": spines_meshes,
        "spines_skeletons": spines_skeletons,
    }

    writer.write_neuron_data(output_file, data)
    writer.write_neuron_meshes(output_file, morph_meshes)


def main() -> None:
    """Main function.

    Returns: None
    """
    parser = argparse.ArgumentParser(
        description="Generate sample collection data in morphology-with-spines format"
    )

    # Output file (string)
    parser.add_argument("-o", "--output", type=str, required=True, help="Output file")

    # Number of neurons (int)
    parser.add_argument("-nneurons", type=int, default=1, help="Number of neurons")

    # Number of collections, if spines grouped by collection only
    parser.add_argument(
        "-ncolls",
        type=int,
        default=0,
        help="Number of spine collections, if spines grouped by collection",
    )

    # Number of spines per collection (int)
    parser.add_argument(
        "-nspines", type=int, default=1, help="Number of spines per collection or per neuron"
    )

    # Grouping by criteria, mutually exclusive: per neuron or per collection
    parser.add_argument("--by-neuron", action="store_true", help="Group spines by neuron")
    parser.add_argument("--by-collection", action="store_true", help="Group spines by collection")

    # FIXME: Not centered option to be implemented, for now they're always centered
    # Centered spines or not (bool)
    # parser.add_argument("--centered", action="store_true", help="Create centered spines")

    args = parser.parse_args()

    if args.by_neuron and args.by_collection:
        raise ValueError("Spines cannot be grouped by neuron and by collection, need to choose one")

    if args.by_neuron and args.ncolls > 0:
        raise ValueError("Number of collections must be 0 if spines are grouped by neuron")

    if args.by_collection and args.ncolls < 1:
        raise ValueError("There must be at least one collection when grouping spines by collection")

    print("Output file:", args.output)
    print("Number of neurons:", args.nneurons)

    if not args.by_collection:
        if not args.by_neuron:
            # Set grouping by neuron by default if no criteria was set
            args.by_neuron = True
            print("Spines grouped by neuron (default)")
        else:
            print("Spines grouped by neuron")

        print("Number of spines per neuron:", args.nspines)
    else:
        print("Spines grouped by collection")
        print("Number of spines per collection:", args.nspines)

    print("Number of spine collections:", args.ncolls)
    print("Spines are centered: True")  # FIXME: To be implemented

    create_sample_data(
        args.output,
        args.nneurons,
        args.ncolls,
        args.nspines,
        args.by_neuron,
        True,  # FIXME: args.centered
    )


if __name__ == "__main__":
    main()
