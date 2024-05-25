import base64
import io
import json
from typing import List

import cv2
import cv2.typing
import numpy as np
from interfaces import Context, Event
from PIL import Image

from src.architectures.segmenter_maskrcnn import MaskRCNNSegmenter
from src.dataset.annotations_coco import COCOAnnotations
from src.dataset.composer import OrderedCompose
from src.dataset.dataset_utils import to_cvat
from src.dataset.preprocessing import CocoPreprocessing
from src.evaluation.prediction import MaskRCNNPrediction

MODEL_PATH = "checkpoint.pth"
CATEGORIES_FILEPATH = "categories.json"
DEVICE = "cuda:0"
BATCH_SIZE = 1


def init_context(context):
    context.logger.info("Init context...  0%")

    num_categories = len(COCOAnnotations.load_file(file_path=CATEGORIES_FILEPATH).get("categories"))
    preprocessing_funcs = OrderedCompose([CocoPreprocessing.resize_to_target], resize_target=1024)

    mask_rcnn = MaskRCNNSegmenter(MODEL_PATH, num_classes=num_categories)
    mask_rcnn.load()
    mask_rcnn = mask_rcnn.to(DEVICE)
    predictor = MaskRCNNPrediction(mask_rcnn, preprocessing_funcs, DEVICE)

    context.user_data.model = predictor
    context.logger.info("Init context...  100%")


def read_categories():
    categories = COCOAnnotations.load_file(file_path=CATEGORIES_FILEPATH).get("categories")
    return {str(category["id"] - 1): category["name"] for category in categories}


def handler(context: Context, event: Event) -> Context.Response:
    context.logger.info("Starting automatic annotation...")

    data = event.body
    buffer = io.BytesIO(base64.b64decode(data["image"]))  # Image to be annotated.
    predictor: MaskRCNNPrediction = context.user_data.model
    threshold = float(data.get("threshold", 0.0))  # Defined on CVAT interface.

    image = np.array(Image.open(buffer))
    rows, cols = image.shape[:2]

    predictions = predictor.predict_image(
        image,
        confidence_threshold=threshold,
        segmentation_threshold=0.5,
        nms_threshold=0.9,
    )

    single_masks = filter_to_single_blob(predictions["masks"])
    resized_masks = []
    discarded_resized_masks = []
    for i in range(len(single_masks)):
        mask = single_masks[i]
        mask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_CUBIC)
        if too_small_to_care(mask, (cols, rows)):
            discarded_resized_masks.append(mask)
        else:
            resized_masks.append(mask)

    print(
        f"total images: {len(single_masks)}; "
        f"considered: {len(resized_masks)}; "
        f"discarded: {len(discarded_resized_masks)}"
    )

    predictions["masks"] = resized_masks
    results = to_cvat(predictions, read_categories())

    context.logger.info("Finished automatic annotation")

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )


def too_small_to_care(mask: cv2.typing.MatLike, size: tuple) -> bool:
    """
    Check if the given mask is too small to be considered for further processing.

    Parameters:
        mask (numpy.ndarray): The mask to be evaluated.
        image_area (int): The total area of the original image.

    Returns:
        bool: True if the mask covers less than 0.05% of the image area, False otherwise.
    """

    image_area = size[0] * size[1]
    mask_area = cv2.countNonZero(mask)
    percentage = (mask_area / image_area) * 100
    return percentage < 0.05


def filter_to_single_blob(masks: List[np.ndarray]) -> List[np.ndarray]:
    """Filter out tiny blobs from a list of masks. Note: The function assumes that the input masks are binary images,
    where non-zero values represent object pixels and zero values represent background pixels.

    Args:
        masks (List[np.ndarray]): A list of numpy arrays representing the masks.

    Returns:
        List[np.ndarray]: A list of numpy arrays representing the masks with tiny blobs filtered out.
    """

    # Due to multiple probability values (pixel level) in the segmentation map, some segmentation might
    # turn to small blobs. However, the better the Mask R-CNN model, the lower the probability of these blobs appear.
    # Here we filter out the tiny blobs, since the first experiments suggest that Mask R-CNN still needs improvement.

    for i in range(len(masks)):
        mask = masks[i]
        n_labels, cc_map = cv2.connectedComponents(mask.astype(np.uint8))

        if n_labels > 2:  # 0 is the background
            blob_sizes = [np.sum(cc_map == i) for i in range(1, n_labels)]
            bigger_blob = np.argmax(blob_sizes) + 1
            mask[cc_map != bigger_blob] = 0
            masks[i] = mask

    return masks
