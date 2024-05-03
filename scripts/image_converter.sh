#!/bin/bash

check_dependencies() {
    # Check if ImageMagick is installed
    if ! command -v convert &> /dev/null; then
        echo "Error: ImageMagick is not installed. Please install it."
        exit 1
    fi
}

check_parameters() {
    # Check if all mandatory parameters are provided
    if [[ -z "$FROM" || -z "$TO" || -z "$IN" ]]; then
        echo "Error: Missing mandatory parameters. Please provide --from, --to, and --in."
        exit 1
    fi

    # Check if --in is a directory
    if [[ ! -d "$IN" ]]; then
        echo "Error: Input directory '$IN' does not exist."
        exit 1
    fi
}

convert_from_to() {
    local FROM="$1"
    local TO="$2"
    local IN="$3"
    local OUT="$4"

    # If --out is not provided, use input directory
    if [[ -z "$OUT" ]]; then
        OUT="$IN"
    fi

    # Convert images
    find "$IN" -type f -print0 | while IFS= read -r -d '' file; do
        filename=$(basename -- "$file")
        extension="${filename##*.}"

        # Check if the file extension matches the specified FROM type
        if [[ "$extension" != "$FROM" ]]; then
            echo "Ignoring (not expected extension) ${filename}"
            continue  # Skip this iteration
        fi
        # Get relative path of file with respect to IN directory
        relative_path=$(realpath --relative-to="$IN" "$file")

        # Create corresponding directory structure in OUT directory
        mkdir -p "$OUT/$(dirname "$relative_path")"

        filename="${filename%.*}"
        convert "$file" "$OUT/$(dirname "$relative_path")/$filename.$TO"
    done

    echo "Conversion completed."
}

# Parse command line parameters
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --from) FROM="$2"; shift ;;
        --to) TO="$2"; shift ;;
        --in) IN="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Check dependencies
check_dependencies

# Check parameters
check_parameters

# Perform conversion
convert_from_to "$FROM" "$TO" "$IN" "$OUT"

