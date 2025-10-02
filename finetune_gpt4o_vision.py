"""
Fine-tune GPT-4o (not mini - vision fine-tuning not supported on mini yet)
to distinguish data center images from non-data center images.

Note: Vision fine-tuning is only available for gpt-4o-2024-08-06 and later,
NOT for gpt-4o-mini.
"""

import os
import json
import base64
import random
from pathlib import Path
from openai import OpenAI
from PIL import Image
from io import BytesIO

# Set API key from classify_sites_concurrent.py
api_key = "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"
client = OpenAI(api_key=api_key)

# Configuration
DATA_CENTER_DIR = "classifier_data/train/data_center"
NON_DATA_CENTER_DIR = "classifier_data/train/non_data_center"
OUTPUT_FILE = "vision_training_data.jsonl"
MODEL = "gpt-4o-2024-08-06"  # Vision fine-tuning supported model

# Limits to avoid file too large error
MAX_IMAGES_PER_CLASS = None  # Use all images
# MAX_IMAGE_SIZE = (512, 512)  # Resize images to reduce file size


def encode_image_to_base64(image_path):
    """Encode image file to base64 string, with resizing to reduce size."""
    # Open and resize image
    img = Image.open(image_path)
    # img.thumbnail(MAX_IMAGE_SIZE, Image.Resampling.LANCZOS)

    # Convert to bytes
    buffer = BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode('utf-8')


def create_training_example(image_path, is_data_center):
    """Create a single training example in OpenAI's format."""
    base64_image = encode_image_to_base64(image_path)
    image_url = f"data:image/png;base64,{base64_image}"

    label = "yes" if is_data_center else "no"

    return {
        "messages": [
            {
                "role": "system",
                "content": "You are a system that helps me with climate planning by identifying whether buildings are data centers."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Is this a data center?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            },
            {
                "role": "assistant",
                "content": label
            }
        ]
    }


def prepare_training_data():
    """Prepare training data from image directories."""
    training_examples = []

    # Process data center images
    data_center_path = Path(DATA_CENTER_DIR)
    if data_center_path.exists():
        all_dc_images = list(data_center_path.glob("*.png"))
        if MAX_IMAGES_PER_CLASS is None:
            sampled_dc_images = all_dc_images
        else:
            sampled_dc_images = random.sample(all_dc_images, min(MAX_IMAGES_PER_CLASS, len(all_dc_images)))
        print(f"Using {len(sampled_dc_images)} of {len(all_dc_images)} data center images")

        for img_file in sampled_dc_images:
            example = create_training_example(str(img_file), is_data_center=True)
            training_examples.append(example)

    # Process non-data center images
    non_data_center_path = Path(NON_DATA_CENTER_DIR)
    if non_data_center_path.exists():
        all_other_images = list(non_data_center_path.glob("*.png"))
        if MAX_IMAGES_PER_CLASS is None:
            sampled_other_images = all_other_images
        else:
            sampled_other_images = random.sample(all_other_images, min(MAX_IMAGES_PER_CLASS, len(all_other_images)))
        print(f"Using {len(sampled_other_images)} of {len(all_other_images)} non-data center images")

        for img_file in sampled_other_images:
            example = create_training_example(str(img_file), is_data_center=False)
            training_examples.append(example)

    # Write to JSONL file
    with open(OUTPUT_FILE, 'w') as f:
        for example in training_examples:
            f.write(json.dumps(example) + '\n')

    print(f"\nTotal training examples: {len(training_examples)}")
    print(f"Training data saved to {OUTPUT_FILE}")
    return OUTPUT_FILE


def upload_and_finetune(training_file):
    """Upload training data and create fine-tuning job."""

    # Upload training file
    print(f"\nUploading {training_file}...")
    with open(training_file, "rb") as f:
        file_response = client.files.create(
            file=f,
            purpose="fine-tune"
        )
    print(f"File uploaded with ID: {file_response.id}")

    # Create fine-tuning job
    print(f"\nCreating fine-tuning job with model {MODEL}...")
    job = client.fine_tuning.jobs.create(
        training_file=file_response.id,
        model=MODEL,
        hyperparameters={
            "n_epochs": 1  # Adjust as needed
        }
    )

    print(f"Fine-tuning job created with ID: {job.id}")
    print(f"Status: {job.status}")
    print(f"\nMonitor your job at: https://platform.openai.com/finetune/{job.id}")
    print(f"Or check status with: client.fine_tuning.jobs.retrieve('{job.id}')")

    return job


def main():
    print("GPT-4o Vision Fine-Tuning for Data Center Detection")
    print("=" * 60)
    print(f"Note: Using {MODEL} (vision fine-tuning NOT available for gpt-4o-mini)\n")

    # Step 1: Prepare training data
    training_file = prepare_training_data()

    # Step 2: Upload and start fine-tuning
    job = upload_and_finetune(training_file)

    print("\n" + "=" * 60)
    print("Fine-tuning initiated successfully!")
    print(f"Once complete, use your fine-tuned model ID: {job.fine_tuned_model or 'will be generated'}")


if __name__ == "__main__":
    main()