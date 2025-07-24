import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch
import io
import pandas as pd

@st.cache_resource
def load_model():
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device

model, feature_extractor, tokenizer, device = load_model()

def generate_caption(image):
    # Preprocess image
    pixel_values = feature_extractor(images=image, return_tensors="pt").pixel_values.to(device)
    # Generate caption ids
    output_ids = model.generate(pixel_values, max_length=40, num_beams=4)
    # Decode caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption

st.title("Alt Text Generator with ViT-GPT2")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg", "bmp", "gif"],
    accept_multiple_files=True,
)

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} images...")
    alt_texts = []

    for uploaded_file in uploaded_files:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
        caption = generate_caption(image)
        alt_texts.append(caption)
        st.write(f"**{uploaded_file.name}**: {caption}")

    df = pd.DataFrame({"Filename": [f.name for f in uploaded_files], "Alt Text": alt_texts})
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "alt_texts.csv", "text/csv")
