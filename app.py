from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

# Initialize BLIP processor and model with use_fast=True for faster processing and to remove warning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Folder containing images - update this path before running
image_folder = "path_to_your_image_folder"
output_file = "alt_texts.csv"

with open(output_file, "w", encoding="utf-8") as f_out:
    f_out.write("image_filename,alt_text\n")

    for filename in os.listdir(image_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(image_folder, filename)
            image = Image.open(image_path).convert('RGB')

            # Prepare inputs
            inputs = processor(image, return_tensors="pt")

            # Generate caption
            out = model.generate(**inputs)
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Write filename and alt text to CSV
            f_out.write(f'"{filename}","{caption}"\n')
            print(f"Processed {filename}: {caption}")

print(f"Alt text generation completed. Results saved in {output_file}")
