import base64
import io
import json

import cv2
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


def handler(context: Context, event: Event) -> Context.Response:
    context.logger.info("Starting automatic annotation...")

    data = event.body
    buffer = io.BytesIO(base64.b64decode(data["image"]))  # Image to be annotated.

    threshold = float(data.get("threshold", 0.5))  # Defined on CVAT interface.
    image = np.array(Image.open(buffer))
    rows, cols = image.shape[:2]

    predictor: MaskRCNNPrediction = context.user_data.model
    predictions = predictor.predict_image(image, threshold)

    resized_masks = []
    for i in range(len(predictions["masks"])):
        mask = predictions["masks"][i]
        mask = cv2.resize(mask, (cols, rows), interpolation=cv2.INTER_NEAREST)
        resized_masks.append(mask)

    predictions["masks"] = resized_masks
    results = to_cvat(predictions)

    context.logger.info("Finished automatic annotation")

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )


if __name__ == "__main__":
    context = Context()
    init_context(context)
    response = handler(context, Event("/home/joaoherrera/data/rugae/manual/images/DSC_0737.JPG", 0.5))
