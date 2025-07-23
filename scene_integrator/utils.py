import cv2
import numpy as np
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_debug(img, path, bgr=True):
    path = Path(path)
    ensure_dir(path.parent)
    if bgr and img.ndim == 3 and img.shape[2] == 3:
        out = img[:, :, ::-1]
    else:
        out = img
    cv2.imwrite(str(path), out)

def to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def reinhard_color_transfer(src_bgr, tgt_bgr, clamp=2.0):
    src = to_lab(src_bgr).astype("float32")
    tgt = to_lab(tgt_bgr).astype("float32")
    for i in range(3):
        sm, ss = src[:, :, i].mean(), src[:, :, i].std() + 1e-6
        tm, ts = tgt[:, :, i].mean(), tgt[:, :, i].std() + 1e-6
        ratio = np.clip(ts / ss, 1 / clamp, clamp)
        src[:, :, i] = (src[:, :, i] - sm) * ratio + tm
    result = np.clip(cv2.cvtColor(src.astype("uint8"), cv2.COLOR_LAB2BGR), 0, 255)
    return result