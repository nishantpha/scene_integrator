import io
import numpy as np
from PIL import Image
from rembg import remove

def extract_person_rgba(person_path: str, model_path: str = "models/u2net.onnx"):
    """
    Return BGRA numpy array (H, W, 4) using rembg (U^2-Net) only.
    """
    with open(person_path, "rb") as f:
        person_bytes = f.read()

    out_bytes = remove(
        person_bytes,
        model="u2net",
        session=None,
        only_mask=False,
        alpha_matting=True,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_structure_size=10,
        alpha_matting_base_size=1000,
    )

    img = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    rgba = np.array(img)
    bgra = rgba[:, :, [2, 1, 0, 3]]  # RGBA -> BGRA
    return bgra