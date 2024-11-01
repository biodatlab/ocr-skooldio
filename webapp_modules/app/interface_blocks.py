import os
import gradio as gr
from .app import WebApp


class BlockInterfaces:
    def __init__(self, model_path: str):
        self.app = WebApp(model_path=model_path)

    def field_detection_interface(self):
        # Show logo if it exists
        if os.path.exists("./logo.png"):
            gr.Image(value="./logo.png", show_label=False, container=False, width=400)

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
            fn=self.app.detect_fields,
            inputs=gr.Image(type="filepath", label="Image"),
            outputs=[
                "image",
                gr.DataFrame(label="Results", headers=["Field", "Text"]),
                gr.File(label="Download Results", file_types=["json"]),
            ],
            allow_flagging="auto",
        )
