import os
import gradio as gr
from webapp_modules.app import WebApp

MODEL_PATH = "models/best.pt"
LOGO_PATH = "logo.png"

assert os.path.exists(
    "models/best.pt"
), "Model not found. Please add the model to the models directory. The model should be named best.pt. -> models/best.pt"
assert os.path.exists(
    LOGO_PATH
), "Logo not found. Please add a logo.png file to the root directory."

demo = gr.Blocks()
webapp = WebApp(model_path="models/best.pt")

with demo:
    with gr.Tab(label="Field Detector"):
        # Show logo if it exists
        gr.Image(value=LOGO_PATH, show_label=False, container=False, width=400)

        gr.Markdown(
            """
            # Field Detector
            A field detector using Optical Character Recognition (OCR) is a technology designed to identify and extract specific data fields from images or scanned documents.
            This type of detector leverages OCR to recognize text within an image and then applies machine learning or rule-based methods to locate and isolate predetermined fields, such as names, dates, addresses, or invoice numbers.
            Field detection is particularly useful in automating data extraction from structured documents like forms, invoices, and receipts.
            It improves efficiency by reducing the need for manual data entry, enhances accuracy by minimizing human error, and provides scalable solutions for processing large volumes of documents across various industries.
            """
        )
        gr.Interface(
            fn=webapp.detect_fields,
            inputs=gr.Image(type="filepath", label="Image"),
            outputs=[
                "image",
                gr.DataFrame(label="Results", headers=["Field", "Text"]),
                gr.File(label="Download Results", file_types=["json"]),
            ],
            allow_flagging="auto",
        )

demo.launch()
