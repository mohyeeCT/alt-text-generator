import streamlit as st
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import io
import torch
import pandas as pd

@st.cache_resource
def load_model():
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
    model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

st.title("Accurate Bulk Alt Text Generator with BLIP-2")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg", "bmp", "gif"],
    accept_multiple_files=True,
)

batch_size = 2  # BLIP-2 is larger, smaller batch to avoid OOM

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} images...")

    images = [Image.open(io.BytesIO(f.read())).convert("RGB") for f in uploaded_files]
    alt_texts = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        inputs = processor(images=batch_images, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_length=40)
        captions = [processor.decode(g, skip_special_tokens=True) for g in generated_ids]
        alt_texts.extend(captions)

    for filename, caption in zip([f.name for f in uploaded_files], alt_texts):
        st.write(f"**{filename}**: {caption}")

    df = pd.DataFrame({"Filename": [f.name for f in uploaded_files], "Alt Text": alt_texts})
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "alt_texts.csv", "text/csv")
