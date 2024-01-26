# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
# Splits a COCO-format dataset into subsets
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#


import os
from argparse import ArgumentParser
from typing import Dict

from src.dataset.dataset_coco import CocoDatasetInstanceSegmentation


def main(args: Dict) -> None:
    """Generates a set of annotations for training, validation, and test splits.

    Args:
        args (Dict): A dictionary containing the arguments for generating annotations.
            annotations_path (str): The path to the annotations file.
            output_path (str): The path where the generated annotations will be saved.
            split (Tuple[int]): A tuple containing the ratios for training, validation, and test splits.
    Raises:
        ValueError: If the sum of the split ratios is not equal to 1.
    """

    annotations_path = args.get("annotations_path")
    output_path = args.get("output_path")
    splits = tuple(args.get("split"))
    split_labels = []

    if sum(splits) != 1:
        raise ValueError("Sum of splits must be 1.")

    match len(splits):
        case 2:
            split_labels = ["training", "validation"]
        case 3:
            split_labels = ["training", "validation", "test"]
        case _:
            split_labels = [f"split_{str(n)}" for n in len(splits)]

    # Load annotations
    coco_dataset = CocoDatasetInstanceSegmentation(data_directory_path=None, data_annotation_path=annotations_path)
    datasets = coco_dataset.split(*splits, random=True)

    os.makedirs(output_path, exist_ok=True)

    for dataset, label in zip(datasets, split_labels):
        dataset.tree.save(os.path.join(output_path, f"{label}_annotations.json"))


def build_arg_parser() -> ArgumentParser:
    """Create and return an argument parser object.

    Returns:
        ArgumentParser: The argument parser object.
    """

    parser = ArgumentParser(description="Split dataset into subsets for training and testing.")

    parser.add_argument(
        "--annotations-path",
        type=str,
        help="Path to the annotation file.",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output folder.",
        required=True,
    )

    parser.add_argument(
        "--split",
        type=float,
        help="Split percentage, e.g. 0.8 0.2.",
        nargs="+",
        required=True,
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    main(vars(args))
