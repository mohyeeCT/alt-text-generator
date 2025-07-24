import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io
import torch

@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

st.title("Bulk Alt Text Generator with GPU & Batching")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg", "bmp", "gif"],
    accept_multiple_files=True,
)

batch_size = 4  # Adjust based on your hardware

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} images...")

    images = [Image.open(io.BytesIO(f.read())).convert("RGB") for f in uploaded_files]
    alt_texts = []

    for i in range(0, len(images), batch_size):
        batch_images = images[i : i + batch_size]
        inputs = processor(batch_images, return_tensors="pt", padding=True).to(device)
        outputs = model.generate(**inputs)
        captions = [processor.decode(o, skip_special_tokens=True) for o in outputs]
        alt_texts.extend(captions)

    # Show results
    for filename, caption in zip([f.name for f in uploaded_files], alt_texts):
        st.write(f"**{filename}**: {caption}")

    # Download CSV
    import pandas as pd
    df = pd.DataFrame({"Filename": [f.name for f in uploaded_files], "Alt Text": alt_texts})
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "alt_texts.csv", "text/csv")
