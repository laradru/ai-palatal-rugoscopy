import argparse
import json
import os
from operator import itemgetter

class Image:
    def __init__(self, id, width, height):
        self.id = id
        self.width = width
        self.height = height

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
            [Image(image['id'], image['width'], image['height']) for image in data['images']],
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

def print_summary(annotations, image_areas):
    # Print smallest annotation info
    print("\nSmallest annotation info:")
    print_annotation_info(annotations.annotations[0], image_areas)

    # Print biggest annotation info
    print("\nBiggest annotation info:")
    print_annotation_info(annotations.annotations[-1], image_areas)

def print_annotation_info(annotation, image_areas):
    print(f"Area: {annotation.area:.2f}")
    print(f"Image area: {image_areas[annotation.image_id]:.2f}")
    print(f"Relative percentage: {(annotation.area / image_areas[annotation.image_id]) * 100:.2f}%")
    print(f"Image ID: {annotation.image_id}")

def print_csv(annotations, image_areas):
    print("Annotation area,Image area,Annotation area percentage,Image ID")
    for ann in annotations.annotations:
        print(f"{ann.area:.2f},{image_areas[ann.image_id]:.2f},{(ann.area / image_areas[ann.image_id]) * 100:.20f},{ann.image_id}")

if __name__ == "__main__":
    main()
