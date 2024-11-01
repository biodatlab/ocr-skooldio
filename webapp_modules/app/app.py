import pandas as pd
from typing import Tuple
from ..detector import CarbookFieldsDetector


class WebApp:
    def __init__(self, model_path: str):
        self.detector = CarbookFieldsDetector(detector_model_path=model_path)

    def detect_fields(
        self, image_path: str, crop_images: bool = True
    ) -> Tuple[str, pd.DataFrame, str]:
        boundaries_image_path, field_texts_dataframe, json_path = self.detector.detect(
            image_path=image_path, crop_images=crop_images
        )
        return boundaries_image_path, field_texts_dataframe, json_path
