import io
import numpy as np
import cv2
from PIL import Image
from rembg import remove
from pymatting import estimate_alpha_cf


def _make_trimap(alpha, fg_thresh=0.98, bg_thresh=0.02, ksize=5):
    """
    Build a trimap from an initial alpha (0..1 float).
    fg_thresh/bg_thresh define sure FG/BG. Unknown in between.
    Erode sure regions slightly to avoid overconfidence.
    """
    fg = (alpha >= fg_thresh).astype(np.uint8) * 255
    bg = (alpha <= bg_thresh).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    fg = cv2.erode(fg, kernel)
    bg = cv2.erode(bg, kernel)

    unknown = 255 - cv2.max(fg, bg)

    trimap = np.zeros_like(alpha, dtype=np.float64)
    trimap[bg == 255] = 0.0
    trimap[fg == 255] = 1.0
    trimap[unknown == 255] = 0.5
    return trimap


def _refine_alpha_cf(image_bgr, alpha_init):
    """
    Refine alpha using Closed-Form Matting (pymatting).
    image_bgr: uint8 HxWx3 (BGR)
    alpha_init: float32 HxW in [0,1]
    returns refined alpha float32 HxW
    """
    trimap = _make_trimap(alpha_init, fg_thresh=0.98, bg_thresh=0.02, ksize=5)
    img_rgb = image_bgr[:, :, ::-1].astype(np.float64) / 255.0  # to RGB float
    alpha_refined = estimate_alpha_cf(img_rgb, trimap)
    alpha_refined = np.clip(alpha_refined, 0.0, 1.0).astype(np.float32)
    return alpha_refined


def extract_person_rgba(person_path: str, model_path: str = "models/u2net.onnx"):
    """
    Return BGRA numpy array (H, W, 4).
    Pipeline:
      1. rembg (U²-Net) for coarse cutout
      2. Closed-form matting (pymatting) to refine edges
    """
    # 1) rembg coarse RGBA
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

    # 2) refine alpha
    bgr = rgba[:, :, :3][:, :, ::-1]  # RGBA->RGB->BGR
    alpha_init = rgba[:, :, 3].astype(np.float32) / 255.0
    alpha_ref = _refine_alpha_cf(bgr, alpha_init)

    rgba[:, :, 3] = (alpha_ref * 255).astype(np.uint8)

    # BGRA for OpenCV consistency
    bgra = rgba[:, :, [2, 1, 0, 3]]
    return bgra