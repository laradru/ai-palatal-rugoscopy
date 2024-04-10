"""Extract patches from a set of images and save them on disk.
"""

from argparse import ArgumentParser
from typing import Dict

from src.dataset.dataset_coco import CocoDatasetInstanceSegmentation


def extract_patches(args: Dict) -> None:
    """Extract patches from the images in the dataset and save them to the output path.

    Args:
        args (Dict): A dictionary containing the arguments for the function.
            - images_path (str): The path to the directory containing the images.
            - annotations_path (str): The path to the annotations file.
            - patch_size (int): The size of the patches to extract.
            - stride (int): The stride between patches.
            - min_area_ratio (float): The minimum area ratio for a patch to be considered valid.
            - output_path (str): The path to save the extracted patches.
    """

    dataset = CocoDatasetInstanceSegmentation(args.get("images_path"), args.get("annotations_path"))
    patch_size = args.get("patch_size")
    stride = args.get("stride")
    min_area_percentage = args.get("min_area_percentage")
    output_path = args.get("output_path")
    resize_image_width = args.get("resize_image_width")

    dataset.extract_patches(output_path, patch_size, stride, min_area_percentage, resize_image_width=resize_image_width)


def build_arg_parser() -> ArgumentParser:
    """Build and return an ArgumentParser object for extracting patches from a set of images and saving them on disk.

    Returns:
        ArgumentParser: The ArgumentParser object that can be used to parse command line arguments.
    """

    parser = ArgumentParser("Extract patches from a set of images and save them on disk.")

    parser.add_argument(
        "--images-path",
        type=str,
        help="Path to a directory containing the images to extract patches.",
        required=True,
    )

    parser.add_argument(
        "--annotations-path",
        type=str,
        help="Path to the annotation file.",
        required=True,
    )

    parser.add_argument(
        "--output-path",
        type=str,
        help="Path to the output directory.",
        required=True,
    )

    parser.add_argument(
        "--resize-image-width",
        type=int,
        help="Width of the resized image. Defaults to None, which means no resizing.",
        default=None,
        required=False,
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        help="Patch size.",
        default=512,
        required=False,
    )

    parser.add_argument(
        "--stride",
        type=int,
        help="Stride.",
        default=1,
        required=False,
    )

    parser.add_argument(
        "--min-area-percentage",
        type=float,
        help="""Patches with foreground area smaller than min_area_percentage pixels in the patch will be discarded.
            Defaults to 0.05.""",
        default=0.05,
        required=False,
    )

    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    extract_patches(vars(args))
