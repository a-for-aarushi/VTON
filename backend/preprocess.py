import cv2
import numpy as np
from PIL import Image
import json
import torch
import io
import base64
from pathlib import Path

def encode_image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def decode_base64_to_image(b64_str: str) -> Image.Image:
    img_bytes = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def resize_and_pad(img: Image.Image, target_size=(768, 1024)) -> Image.Image:
    """Resize image to VITON-HD expected dimensions"""
    return img.resize(target_size, Image.LANCZOS)

def generate_cloth_mask(cloth_img: Image.Image) -> Image.Image:
    """
    Auto-generate cloth mask using GrabCut or simple thresholding.
    For best results, use a white background cloth image.
    """
    img_np = np.array(cloth_img)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    
    # Threshold: assume white/light background
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return Image.fromarray(mask)

def run_openpose(person_img: Image.Image):
    """
    Run OpenPose to get keypoints.
    Returns pose image and pose JSON.
    In Modal, we'll use a lightweight alternative.
    """
    # Placeholder - actual OpenPose runs in Modal container
    pass

def prepare_inputs_for_viton(person_img: Image.Image, cloth_img: Image.Image):
    """Prepare all inputs needed by VITON-HD"""
    person_resized = resize_and_pad(person_img, (768, 1024))
    cloth_resized = resize_and_pad(cloth_img, (768, 1024))
    cloth_mask = generate_cloth_mask(cloth_resized)
    
    return {
        "person": person_resized,
        "cloth": cloth_resized,
        "cloth_mask": cloth_mask,
    }