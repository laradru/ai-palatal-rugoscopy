import base64
from dataclasses import dataclass


class Logger:
    def info(self, message: str) -> None: ...  # noqa
    def error(self, message: str) -> None: ...  # noqa


class UserData:
    model = None


@dataclass
class Response:
    body: str
    headers: dict
    content_type: str
    status_code: int


class Context:
    logger = Logger()
    user_data = UserData()
    Response = Response


class Event:
    def __init__(self, image_path: str, threshold: float) -> None:
        """Initialize the object with the provided image path and threshold.

        Parameters:
            image_path (str): The path to the image file.
            threshold (float): The threshold value.
        """

        with open(image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("ascii")
            self.body = {"image": image_base64, "threshold": threshold}
