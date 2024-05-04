import argparse
import json


class Image:
    def __init__(self, id, width, height, filename):
        self.id = id
        self.width = width
        self.height = height
        self.filename = filename


class Annotation:
    def __init__(self, area, image_id):
        self.area = area
        self.image_id = image_id


class Annotations:
    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations


def main():
    # Define command-line arguments
    parser = argparse.ArgumentParser(description='Process JSON file containing annotations.')
    parser.add_argument('json_file', metavar='JSON_FILE', type=str, help='Path to JSON file')
    parser.add_argument('--csv', action='store_true', help='Print comma-separated values')
    parser.add_argument('--summary', action='store_true', help='Include summary at the end')
    parser.add_argument('--image_info', action='append', type=int,
                        help='Show details for an image ID (can be specified multiple times)')
    args = parser.parse_args()

    # Read JSON file
    try:
        with open(args.json_file, 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        print("Error: File not found.")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format.")
        return

    # Parse JSON data
    try:
        annotations = Annotations(
            [Image(image['id'], image['width'], image['height'], image['file_name']) for image in data['images']],
            [Annotation(annotation['area'], annotation['image_id']) for annotation in data['annotations']]
        )
    except KeyError:
        print("Error: JSON data missing required fields.")
        return

    # Sort annotations by area
    annotations.annotations.sort(key=lambda x: x.area)

    # Calculate image areas
    image_areas = {image.id: image.width * image.height for image in annotations.images}

    # Print CSV format if requested
    if args.csv:
        print_csv(annotations, image_areas)

    # Print summary if requested
    if args.summary:
        print_summary(annotations, image_areas)

    # Process image IDs if provided
    if args.image_info:
        process_image_ids(args.image_info, annotations)


def process_image_ids(image_ids, annotations):
    for image_id in image_ids:
        print_image_info(annotations, image_id)


def print_image_info(annotations, image_id):
    found = False
    for image in annotations.images:
        if image.id == image_id:
            print(f"Image ID: {image.id}")
            print(f"Image width: {image.width}")
            print(f"Image height: {image.height}")
            print(f"Image filename: {image.filename}")
            print()
            found = True
            break
    if not found:
        print(f"Image ID {image_id} not found.")
        print()


def print_summary(annotations, image_areas):
    # Print smallest annotation info
    print("\nSmallest annotation info:")
    print_annotation_info(annotations.annotations[0], image_areas)
    print_image_info(annotations, annotations.annotations[0].image_id)

    # Print biggest annotation info
    print("Biggest annotation info:")
    print_annotation_info(annotations.annotations[-1], image_areas)
    print_image_info(annotations, annotations.annotations[-1].image_id)


def print_annotation_info(annotation, image_areas):
    print(f"Area: {annotation.area:.2f}")
    print(f"Image area: {image_areas[annotation.image_id]:.2f}")
    print(f"Relative percentage: {(annotation.area / image_areas[annotation.image_id]) * 100:.2f}%")
    print(f"Image ID: {annotation.image_id}")


def print_csv(annotations, image_areas):
    print("Annotation area,Image area,Annotation area percentage,Image ID")
    for ann in annotations.annotations:
        print(
            f"{ann.area:.2f},{image_areas[ann.image_id]:.2f},{(ann.area / image_areas[ann.image_id]) * 100:.20f},{ann.image_id}")


if __name__ == "__main__":
    main()
