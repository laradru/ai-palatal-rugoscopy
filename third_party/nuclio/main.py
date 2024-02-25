import base64
import io
import json

import numpy as np
from interfaces import Context, Event
from PIL import Image

from src.dataset.annotations_coco import COCOAnnotations
from src.evaluation.prediction import MaskRCNNPrediction

MODEL_PATH = "checkpoint.pth"
CATEGORIES_FILEPATH = "categories.json"
PATCH_SIZE = 256
STRIDE = 128
DEVICE = "cuda:0"


def init_context(context):
    context.logger.info("Init context...  0%")

    num_categories = len(COCOAnnotations.load_file(file_path=CATEGORIES_FILEPATH).get("categories"))
    predictor = MaskRCNNPrediction(MODEL_PATH, num_categories, PATCH_SIZE, STRIDE, DEVICE)

    context.user_data.model = predictor
    context.logger.info("Init context...  100%")


def handler(context: Context, event: Event) -> Context.Response:
    data = event.body
    buffer = io.BytesIO(base64.b64decode(data["image"]))  # Image to be annotated.
    threshold = float(data.get("threshold", 0.5))  # Defined on CVAT interface.

    image = np.array(Image.open(buffer))
    predictor: MaskRCNNPrediction = context.user_data.model

    results = predictor.predict_image(image, threshold)
    results = predictor.to_cvat(results)

    return context.Response(
        body=json.dumps(results),
        headers={},
        content_type="application/json",
        status_code=200,
    )
