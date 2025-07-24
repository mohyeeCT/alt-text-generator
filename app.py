import streamlit as st
import openai
import base64
import io

# Set your OpenAI API key
openai.api_key = st.secrets["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"]

st.title("Alt Text Generator with GPT-4 Vision")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "gif"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Convert image to base64 for prompt
    image_bytes = uploaded_file.read()
    encoded_image = base64.b64encode(image_bytes).decode()

    prompt = (
        "Describe the following image in a detailed way suitable for alt text:\n"
        f"data:image/jpeg;base64,{encoded_image}"
    )

    if st.button("Generate Alt Text"):
        with st.spinner("Generating alt text..."):
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            alt_text = response['choices'][0]['message']['content']
            st.text_area("Generated Alt Text", value=alt_text, height=150)
