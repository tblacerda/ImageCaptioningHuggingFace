import os
import glob
import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration #Blip2 models

# Load the pretrained BLIP-2 processor and model from Hugging Face
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")

# Specify the directory containing the images
image_dir = "/home/tiago/Imagens"
# List of image file extensions to look for
image_exts = ["jpg", "jpeg", "png"]

# Open a text file to write the generated captions
with open("captions.txt", "w") as caption_file:
    # Loop through each image extension
    for image_ext in image_exts:
        # Find all image files in the directory with the current extension
        for img_path in glob.glob(os.path.join(image_dir, f"*.{image_ext}")):
            # Open the image and convert it to RGB format
            raw_image = Image.open(img_path).convert('RGB')

            # Preprocess the image for the model
            inputs = processor(raw_image, return_tensors="pt")

            # Generate a caption for the image using the model
            out = model.generate(**inputs, max_new_tokens=50)

            # Decode the generated output tokens to a human-readable string
            caption = processor.decode(out[0], skip_special_tokens=True)

            # Write the image filename and its caption to the output file
            caption_file.write(f"{os.path.basename(img_path)}: {caption}\n")