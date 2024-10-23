import io
from collections.abc import Sequence

import numpy as np
import streamlit as st
from easyocr import Reader
from PIL import Image, ImageDraw
from PIL.ImageOps import exif_transpose

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class TextReader:
    """A class to read text from an image."""

    def __init__(self, languages: Sequence[str] = ("en", "th")) -> None:
        """Initialize the TextReader object."""
        # By default, we use English and Thai languages
        self.reader = Reader(languages)

    def detect(
        self, image: Image.Image | np.ndarray
    ) -> list[tuple[int, int, int, int]]:
        """Detect text in the image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Perform text detection
        # `horizontal_list` is a list of bounding boxes for horizontal text e.g. `[x_min, x_max, y_min, y_max]`.
        # `free_list` is a list of bounding boxes for free-form text e.g. `[[x1,y1],[x2,y2],[x3,y3],[x4,y4]]`.
        horizontal_list, free_list = self.reader.detect(image)
        # Get the first result because we only input one image.
        bboxes = horizontal_list[0]
        # Convert to the format `[left, top, right, bottom]`
        bboxes = [(bbox[0], bbox[2], bbox[1], bbox[3]) for bbox in bboxes]
        return bboxes

    def read(self, image: Image.Image | np.ndarray) -> tuple[str, float]:
        """Read text in the image."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        # Get the first result because we only input one image.
        result = self.reader.recognize(image)[0]
        # Unpack the result
        _, text, confidence_score = result
        return text, confidence_score


# Set the page title and icon
st.set_page_config(
    page_title="OCR Skooldio",
    page_icon="👀",
)

# Write the app title and description
st.title("OCR Skooldio")
st.write("A webapp for inferencing OCR model.")

##################
# Image uploader #
##################

# Display the file uploader and toggle button
file = st.file_uploader("Upload an image", type=SUPPORTED_IMAGE_EXTENSIONS)

# Load image if the user has uploaded a file
if file is None:
    st.stop()

image = Image.open(io.BytesIO(file.read())).convert("RGB")
# Rotate the image based on the EXIF data
image = exif_transpose(image)

# Display the uploaded image
st.subheader("Uploaded Image")
st.image(image, use_column_width=True)

recognition_only = st.checkbox(
    "Recognition Only", value=True, help="Only recognize text in the image"
)

classify_button = st.button("Run")

###############
# Predictions #
###############

# Display the result when the user clicks the "Classify" button
if not classify_button:
    st.stop()

# Load the model in the session state if it does not exist
if "reader" not in st.session_state:
    with st.spinner("Loading the model..."):
        st.session_state["reader"] = TextReader()

reader: TextReader = st.session_state["reader"]

if file is None:
    st.error("Please upload an image.")
    st.stop()

# Add a divider to separate the result from the button

if not recognition_only:
    # Perform text detection first.
    with st.spinner("Detecting texts..."):
        bboxes = reader.detect(image)
    # Check if there are any textboxes detected
    if len(bboxes) == 0:
        st.error("No textboxes detected.")
        st.stop()
    # Cropping textboxes
    with st.spinner("Cropping textboxes..."):
        textboxes = [image.crop(bbox) for bbox in bboxes]
else:
    textboxes = [image]

# Show bounding box result if the user chooses to detect text
if not recognition_only:
    st.divider()
    st.subheader("Bounding Boxes:")
    display_image = image.copy()
    drawing = ImageDraw.Draw(display_image)
    for bbox in bboxes:
        drawing.rectangle(bbox, outline="red", width=3)
    st.image(display_image, use_column_width=True)

# Perform text recognition
progress_text = "Recognizing textboxes..."
progress_bar = st.progress(0, text=progress_text)
total_textboxes = len(textboxes)
texts, scores = [], []
for idx, textbox in enumerate(textboxes):
    text, score = reader.read(textbox)
    texts.append(text)
    scores.append(score)
    progress_bar.progress((idx + 1) / total_textboxes, text=progress_text)
progress_bar.empty()

# If recognition only, just display the result
if recognition_only:
    for text, score in zip(texts, scores):
        # Display the result
        color = "green" if score > 0.9 else "red"
        col1, col2 = st.columns([1, 5])
        col1.metric(f":{color}[Confidence]", round(score, 4))
        col2.text(text)
# Display bounding boxes too
else:
    st.divider()
    st.subheader("Predictions:")

    for text, textbox, score in zip(texts, textboxes, scores):
        # Display the textbox
        st.image(textbox, use_column_width=True)
        # Display the result
        color = "green" if score > 0.9 else "red"
        col1, col2 = st.columns([1, 5])
        col1.metric(f":{color}[Confidence]", round(score, 4))
        col2.text(text)
        st.divider()
