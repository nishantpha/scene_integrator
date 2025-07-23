import io
import numpy as np
from PIL import Image
from rembg import remove
from .matting_fba import load_fba, refine_alpha_fba

_fba_model = None  # lazy load

def extract_person_rgba(person_path: str,
                        model_path: str = "models/u2net.onnx",
                        use_fba: bool = True,
                        fba_weights: str = "models/fba_matting.pth"):
    """
    Returns BGRA numpy array (H, W, 4). Uses rembg for coarse cut & optional FBA matting refinement.
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
    img_rgba = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
    rgba = np.array(img_rgba).astype(np.uint8)

    if use_fba:
        global _fba_model
        if _fba_model is None:
            _fba_model = load_fba(fba_weights)

        # RGBA -> BGR + initial alpha
        bgr = rgba[:, :, :3][:, :, ::-1]  # RGB -> BGR
        alpha_init = rgba[:, :, 3].astype(np.float32) / 255.0
        alpha_refined = refine_alpha_fba(bgr, alpha_init, _fba_model)
        rgba[:, :, 3] = (alpha_refined * 255).astype(np.uint8)

    # return BGRA
    bgra = rgba[:, :, [2, 1, 0, 3]]
    return bgra