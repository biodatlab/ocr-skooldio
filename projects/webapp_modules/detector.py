import json
import os
import torch
import pandas as pd
import numpy as np
from skimage.color import rgb2gray
from skimage.transform import rotate
from deskew import determine_skew
from ultralytics import YOLO
from PIL import Image, ImageDraw
from typing import Tuple
from surya.recognition import batch_recognition
from surya.model.recognition.model import load_model as load_recognizer
from surya.model.recognition.processor import (
    load_processor as load_recognizer_processor,
)


class CarbookFieldsDetector:
    def __init__(self, detector_model_path: str):
        # Load a model from a checkpoint
        self.model = YOLO(detector_model_path)
        self.base_output_path = "./temp"
        self.image_output_path = ""
        self.recognizer = load_recognizer()
        self.recognizer_processor = load_recognizer_processor()

    @staticmethod
    def deskew_image(image: Image) -> Image:
        """

        """
        # Convert image to numpy array
        image = np.array(image)
        # Convert image to grayscale
        grey_image = rgb2gray(image)
        angle = determine_skew(grey_image)

        rotated = rotate(image, angle, resize=True) * 255 # Rotate the image
        return Image.fromarray(rotated.astype("uint8"))

    def detect(self, image_path: str, deskew_image: bool, crop_image: bool) -> Image:
        """
        Detects cars and books in an image
        :param image_path: The path to the image
        :param crop_images: Whether to crop the detected objects
        :return: The path to the output image
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load image
        image = Image.open(image_path)

        # Deskew image
        if deskew_image:
            image = self.deskew_image(image)

        # Predict
        result = self.model.predict(
            image, imgsz=640, conf=0.25, half=True, device=device
        )
        result = result[0]

        # Get image and draw boxes
        image = Image.fromarray(result.orig_img)
        # Convert image to RGB to regain the original colors
        b, g, r = image.split()
        image = Image.merge("RGB", (r, g, b))
        # Get bounding boxes, class labels, and scores
        class_names = result.names
        bboxes = result.boxes.data[:, :4].round().cpu().int().numpy().tolist()
        class_predictions = result.boxes.cls.cpu().int().numpy()

        # Create output directory
        image_name = os.path.basename(image_path)

        output_path = os.path.join(self.base_output_path, image_name)
        self.image_output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        crop_dir = os.path.join(output_path, "crops")
        os.makedirs(crop_dir, exist_ok=True)

        fields = {}
        # Draw bounding boxes and labels on image
        for bbox, cls in zip(bboxes, class_predictions):
            # Draw bounding box and label
            x1, y1, x2, y2 = bbox
            predicted_class = class_names[cls]
            # Add bounding box to fields dictionary
            fields[predicted_class] = {"boundaries": [x1, y1, x2, y2]}
            # Draw bounding box
            draw = ImageDraw.Draw(image)
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)

            # Crop image based on bounding box
            if crop_image:
                cropped_image = image.crop((x1, y1, x2, y2))
                predicted_name = predicted_class.replace(" ", "_").replace("/", "_")
                cropped_path = os.path.join(crop_dir, f"{predicted_name}.jpg")
                cropped_image.save(cropped_path)
                fields[predicted_class]["crop_path"] = cropped_path

        # Save image
        drawn_boundaries_image_path = os.path.join(output_path, f"{image_name}.png")
        image.save(drawn_boundaries_image_path)

        # Note: the fields dictionary is a placeholder for the actual extracted fields
        # Overwrite the fields dictionary and output path with the new values
        self.fields = fields
        self.output_path = output_path
        field_texts, json_path = self.recognise_texts()

        return drawn_boundaries_image_path, field_texts, json_path

    def recognise_texts(self) -> Tuple[pd.DataFrame, str]:
        """
        Recognize texts from the cropped images
        :return: A DataFrame of the extracted fields and the path to the JSON file
        """
        # List crop paths
        crop_paths = [
            self.fields[field]["crop_path"]
            for field in self.fields
            if "crop_path" in self.fields[field]
        ]
        pil_images = [Image.open(crop_path) for crop_path in crop_paths]
        # Recognize texts
        predicted_texts = batch_recognition(
            images=pil_images,
            languages=[["th", "en"]] * len(pil_images),
            model=self.recognizer,
            processor=self.recognizer_processor,
            batch_size=4,
        )[0]

        for text, field in zip(predicted_texts, self.fields):
            # A little bit of cleaning
            text = " ".join(text.split())  # Remove extra whitespaces
            text = text.strip()  # Remove leading and trailing

            self.fields[field]["text"] = text

        extracted_fields = self.fields.keys()
        extracted_texts = [self.fields[field]["text"] for field in extracted_fields]

        field_texts = pd.DataFrame({"field": extracted_fields, "text": extracted_texts})
        json_fields = self.fields
        # Removed crop paths
        for field in json_fields:
            json_fields[field].pop("crop_path", None)

        # Export to JSON
        json_path = os.path.join(self.output_path, "fields.json")
        with open(json_path, "w") as f:
            json.dump(json_fields, f, ensure_ascii=False, indent=4)

        return field_texts, json_path
