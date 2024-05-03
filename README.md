# AI Palatal Rugoscopy

Instance segmentation of palatal rugae patterns using Deep Learning.

## Development environment

By default, we use a [Docker](https://www.docker.com/) container as the development environment.
Before building the image, please ensure you have the following resources available:

- NVIDIA GPU with an updated driver version
- `nvidia-container-tookit` (allows running GPU-accelerated containers)

The script [`docker/build-container.sh`](docker/build-container.sh) automatically builds the image and creates the container. To execute this script, run the following command in a bash terminal:

```shell
cd docker
bash build_container.sh --data-path <DATA_PATH>
```

Where:

- DATA_PATH is the path to a folder containing the dataset (images).

## Image convertion script

There's a script that might be helpful if you need to convert images. In our case, from HEIC to something else that CVAT supports. 

Use it as such:

```shell
./scripts/image_converter.sh --from heic --to jpg --in /path/to/input/directory [--out /path/to/output/directory]
```

## Dataset Preprocessing

There are a few scripts available to preprocess data. They can be found in the [src/standalone](src/standalone/) directory.

- `dataset_split.py`: Splits a dataset into `train`, `test` and `validation` sets. Usage: 
  ```shell
  python dataset_split.py --annotations-path <ANNOTATION_FILE> --output-path <OUTPUT_DIRECTORY> --split <TRAIN_SIZE> <VALIDATION_SIZE> [<TEST_SIZE>]
  ```

    Where

  - ANNOTATION_FILE: Path to annotation file.
  - OUTPUT_DIRECTORY: Path to the output file.
  - TRAIN_SIZE: Percentage of samples to be allocated in the train subset.
  - VALIDATION_SIZE: Percentage of samples to be allocated in the validation subset.
  - TEST_SIZE (Optional): Percentage of samples to be allocated in the test subset.

- `dataset_patches_generation.py`: Break images into patches. Usage:
  ```shell
  python dataset_patches_generation.py --input-directory <INPUT_DIRECTORY> --input-annotations <INPUT_ANNOTATIONS> --output-directory <OUTPUT_DIRECTORY>
  ```

    Where

  - INPUT_DIRECTORY: Path to the directory containing images with annotations.
  - INPUT_ANNOTATIONS: Path to the images' annotation file.
  - OUTPUT_DIRECTORY: Path to the output directory.

## Model Training & Evaluation

Training a Deep Learning model for a given computer vision task is taken care of by
[train.py](src/training/train.py). It can be executed as follows:

```shell
python src.training.train \
    --training-images <TRAINING_IMAGES_DIRECTORY> \
    --training-annotations <TRAINING_ANNOTATIONS_PATH> \
    --validation-images <VALIDATION_IMAGES_DIRECTORY> \
    --validation-annotations <VALIDATION_ANNOTATIONS_PATH> \
    --output-path <OUTPUT_DIRECTORY> \
    --batch-size <BATCH_SIZE> \
    --epochs <EPOCHS> \
    --learning-rate <LEARNING_RATE> \
    --seed <SEED> \
    <--preprocess> \
    <--augment> \
    <--gpu> 
```

Where:

- TRAINING_IMAGES_DIRECTORY: Path to the directory containing training images.
- TRAINING_ANNOTATIONS_PATH: Path to the annotation file of the training set.
- VALIDATION_IMAGES_DIRECTORY: Path to the directory containing validation images.
- VALIDATION_ANNOTATIONS_PATH: Path to the annotation file of the validation set.
- OUTPUT_DIRECTORY: Path to the output directory, where the model will be saved.
- BATCH_SIZE: Size of training and validation batches. Default 16.
- EPOCHS: Number of epochs. Default 50.
- LEARNING_RATE: Learning rate. Defaults to 0.001.
- SEED: A seed for reproducibility. Defaults to 2183648025.
- preprocess: Whether to apply preprocessing algorithms (hard-coded in the train script for now).
- augment: Whether to augment data using Albumentations (check `src/dataset/augmentations.py` for more details).
- gpu: Whether to use GPU, otherwise CPU.

We also provide a [Jupyter notebook](src/evaluation/evaluation.ipynb) for model evaluation, so one can quickly check quantitative and qualitative metrics as well as export them to a PDF file.
