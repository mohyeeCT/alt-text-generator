import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import io

# Load model and processor once
@st.cache_resource
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

processor, model = load_model()

st.title("Bulk Alt Text Generator")

uploaded_files = st.file_uploader("Upload images", type=["png", "jpg", "jpeg", "bmp", "gif"], accept_multiple_files=True)

if uploaded_files:
    st.write(f"Processing {len(uploaded_files)} images...")
    results = []

    for uploaded_file in uploaded_files:
        image = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")

        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)

        results.append((uploaded_file.name, caption))

    # Show results in a table
    st.write("### Generated Alt Texts")
    for filename, alt_text in results:
        st.write(f"**{filename}**: {alt_text}")

    # Option to download CSV
    import pandas as pd
    df = pd.DataFrame(results, columns=["Filename", "Alt Text"])
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "alt_texts.csv", "text/csv")
