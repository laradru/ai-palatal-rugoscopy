.PHONY = patch train

CVAT_EXPORT_ROOT = 1

PATCH_IMAGES_PATH = ../../data/${CVAT_EXPORT_ROOT}/images
PATCH_ANNOTATIONS_FILENAME = instances_default.json
PATCH_ANNOTATIONS_PATH = ../../data/${CVAT_EXPORT_ROOT}/annotations/${PATCH_ANNOTATIONS_FILENAME}
PATCH_OUTPUT_PATH = ../../data/${CVAT_EXPORT_ROOT}/output/patches
PATCH_SIZE = 256
PATCH_STRIDE = 128

patch:
    python -m src.extras.dataset_patch_extraction \
    --images-path ${PATCH_IMAGES_PATH} \
    --annotations-path ${PATCH_ANNOTATIONS_PATH} \
    --output-path ${PATCH_OUTPUT_PATH} \
    --patch-size ${PATCH_SIZE} \
    --stride ${PATCH_STRIDE}

TRAIN_IMAGES_PATH = ../../data/${CVAT_EXPORT_ROOT}/output/patches/images
TRAIN_ANNOTATIONS_FILENAME = annotations.json
TRAIN_ANNOTATIONS_PATH = ../../data/${CVAT_EXPORT_ROOT}/output/patches/annotations/${TRAIN_ANNOTATIONS_FILENAME}
TRAIN_VALIDATION_IMAGES_PATH = ${TRAIN_IMAGES_PATH}
TRAIN_VALIDATION_ANNOTATIONS_PATH = ${TRAIN_ANNOTATIONS_PATH}
TRAIN_OUTPUT_PATH = ../../data/${CVAT_EXPORT_ROOT}/output/train/${PATCH_SIZE}/
TRAIN_BATCH_SIZE = 2

train:
    python -m src.training.train \
    --training-images ${TRAIN_IMAGES_PATH} \
    --training-annotations ${TRAIN_ANNOTATIONS_PATH} \
    --validation-images ${TRAIN_VALIDATION_IMAGES_PATH} \
    --validation-annotations ${TRAIN_VALIDATION_ANNOTATIONS_PATH} \
    --output-path ${TRAIN_OUTPUT_PATH} \
    --batch-size ${TRAIN_BATCH_SIZE} \
    --gpu
