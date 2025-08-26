import numpy as np
from PIL import Image, ImageFilter

def resize_image(image, size=(224, 224)):
    """Resize gambar untuk CNN."""
    return image.resize(size)

def normalize_image(image):
    """Normalisasi pixel 0–1 lalu kembalikan lagi sebagai PIL.Image."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

def apply_filter(image):
    """Simulasi konvolusi dengan filter edge detection."""
    return image.filter(ImageFilter.FIND_EDGES)

def resize_yolo(image, size=(640, 640)):
    """Resize gambar untuk YOLO."""
    return image.resize(size)

def normalize_yolo(image):
    """Normalisasi pixel 0–1 lalu kembalikan lagi sebagai PIL.Image."""
    img_array = np.array(image, dtype=np.float32) / 255.0
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)
