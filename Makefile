.PHONY = patch train

DATA_ROOT = ../../data/rugae
CVAT_EXPORT_ROOT = manual

SPLIT_ANNOTATIONS_FILENAME = instances_default.json
SPLIT_ANNOTATIONS_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/annotations/${SPLIT_ANNOTATIONS_FILENAME}
SPLIT_OUTPUT_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/output/
SPLIT_SPLIT = 0.8 0.2

split:
	python -m src.extras.dataset_split \
	--annotations-path ${SPLIT_ANNOTATIONS_PATH} \
	--output-path ${SPLIT_OUTPUT_PATH} \
	--split ${SPLIT_SPLIT}

PATCH_IMAGES_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/images
PATCH_ANNOTATIONS_TRAINING_FILENAME = training_annotations.json
PATCH_ANNOTATIONS_TRAINING_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/output/${PATCH_ANNOTATIONS_TRAINING_FILENAME}
PATCH_ANNOTATIONS_VALIDATION_FILENAME = validation_annotations.json
PATCH_ANNOTATIONS_VALIDATION_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/output/${PATCH_ANNOTATIONS_VALIDATION_FILENAME}
PATCH_TRAINING_OUTPUT_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/output/patches/training
PATCH_VALIDATION_OUTPUT_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/output/patches/validation
PATCH_RESIZE_IMAGE_WIDTH = 1024
PATCH_SIZE = 256
PATCH_STRIDE = 16

patch:
	python -m src.extras.dataset_patch_extraction \
    --images-path ${PATCH_IMAGES_PATH} \
    --annotations-path ${PATCH_ANNOTATIONS_TRAINING_PATH} \
    --output-path ${PATCH_TRAINING_OUTPUT_PATH} \
    --patch-size ${PATCH_SIZE} \
    --stride ${PATCH_STRIDE} \
    --resize-image-width ${PATCH_RESIZE_IMAGE_WIDTH} && \
    python -m src.extras.dataset_patch_extraction \
    --images-path ${PATCH_IMAGES_PATH} \
    --annotations-path ${PATCH_ANNOTATIONS_VALIDATION_PATH} \
    --output-path ${PATCH_VALIDATION_OUTPUT_PATH} \
    --patch-size ${PATCH_SIZE} \
    --stride ${PATCH_STRIDE} \
    --resize-image-width ${PATCH_RESIZE_IMAGE_WIDTH}
    

TRAIN_TRAINING_IMAGES_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/images
TRAIN_TRAINING_ANNOTATIONS_FILENAME = training_annotations.json
TRAIN_TRAINING_ANNOTATIONS_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/annotations/${TRAIN_TRAINING_ANNOTATIONS_FILENAME}
TRAIN_VALIDATION_IMAGES_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/images
TRAIN_VALIDATION_ANNOTATIONS_FILENAME = validation_annotations.json
TRAIN_VALIDATION_ANNOTATIONS_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/annotations/${TRAIN_VALIDATION_ANNOTATIONS_FILENAME}
TRAIN_OUTPUT_PATH = ${DATA_ROOT}/${CVAT_EXPORT_ROOT}/output/train/full
TRAIN_COCO_EVAL_FREQUENCY = 3
TRAIN_BATCH_SIZE = 3
TRAIN_EPOCHS = 10
TRAIN_LEARNING_RATE = 0.001

train:
	python -m src.training.train \
    --training-images ${TRAIN_TRAINING_IMAGES_PATH} \
    --training-annotations ${TRAIN_TRAINING_ANNOTATIONS_PATH} \
    --validation-images ${TRAIN_VALIDATION_IMAGES_PATH} \
    --validation-annotations ${TRAIN_VALIDATION_ANNOTATIONS_PATH} \
    --output-path ${TRAIN_OUTPUT_PATH} \
    --batch-size ${TRAIN_BATCH_SIZE} \
    --learning-rate ${TRAIN_LEARNING_RATE} \
    --coco-eval-frequency ${TRAIN_COCO_EVAL_FREQUENCY} \
    --epochs ${TRAIN_EPOCHS} \
    --preprocess \
    --gpu
