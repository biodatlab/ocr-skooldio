import gradio as gr
from fastapi import FastAPI
from webapp_modules.app.interface_blocks import BlockInterfaces

app_blocks = BlockInterfaces(model_path="models/best.pt")
app = FastAPI()

demo = gr.Blocks()

with demo:
    with gr.Tab(label="Field Detector"):
        app_blocks.field_detection_interface()

demo.launch()
