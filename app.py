import streamlit as st
from google.cloud import vision
from google.oauth2 import service_account
from PIL import Image
import io
import json
import pandas as pd

# Step 1: Upload the service account JSON key
key_file = st.file_uploader("Upload Google Cloud Service Account JSON Key", type=["json"])

client = None
if key_file is not None:
    # Step 2: Load credentials from uploaded file
    key_json = json.load(key_file)
    credentials = service_account.Credentials.from_service_account_info(key_json)
    client = vision.ImageAnnotatorClient(credentials=credentials)
    st.success("Google Cloud Vision client initialized.")

def generate_alt_text(image_bytes, client):
    image = vision.Image(content=image_bytes)
    response = client.label_detection(image=image)
    labels = response.label_annotations

    if response.error.message:
        st.error(f"API error: {response.error.message}")
        return "Error generating alt text"

    alt_text = ', '.join(label.description for label in labels[:5])
    return alt_text

st.title("Alt Text Generator with Google Cloud Vision API")

uploaded_files = st.file_uploader(
    "Upload images",
    type=["png", "jpg", "jpeg", "bmp", "gif"],
    accept_multiple_files=True,
)

if uploaded_files and client is not None:
    st.write(f"Processing {len(uploaded_files)} images...")
    alt_texts = []

    for uploaded_file in uploaded_files:
        image_bytes = uploaded_file.read()
        alt_text = generate_alt_text(image_bytes, client)
        alt_texts.append(alt_text)
        st.write(f"**{uploaded_file.name}**: {alt_text}")

    df = pd.DataFrame({"Filename": [f.name for f in uploaded_files], "Alt Text": alt_texts})
    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "alt_texts.csv", "text/csv")

elif key_file is None:
    st.info("Please upload your Google Cloud service account JSON key to get started.")
