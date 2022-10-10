import gradio as gr
from transformers import pipeline
import spacy

spacy.cli.download('en_core_web_sm')

pipeline = pipeline(task="image-classification", model="julien-c/hotdog-not-hotdog")


def predict(image, threshold=0.4):
  predictions = pipeline(image)
  return {p["label"]: p["score"] for p in predictions}


gr.Interface(
    predict,
    inputs=['text', gr.Slider(0, 1.0, value=0.4, step=0.05)],
    outputs=gr.outputs.Label(num_top_classes=2),
    title='What the DAAM!?',
).launch()
