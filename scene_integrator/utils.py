import cv2
import numpy as np
from pathlib import Path

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_debug(img, path, bgr=True):
    path = Path(path)
    ensure_dir(path.parent)
    out = img[:, :, ::-1] if (bgr and img.ndim==3 and img.shape[2]==3) else img
    cv2.imwrite(str(path), out)

def to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def reinhard_color_transfer(src_bgr, tgt_bgr):
    src = to_lab(src_bgr).astype("float32")
    tgt = to_lab(tgt_bgr).astype("float32")
    for i in range(3):
        src_mean, src_std = src[:, :, i].mean(), src[:, :, i].std()
        tgt_mean, tgt_std = tgt[:, :, i].mean(), tgt[:, :, i].std()
        src[:, :, i] = (src[:, :, i] - src_mean) * (tgt_std / (src_std + 1e-6)) + tgt_mean
    result = np.clip(cv2.cvtColor(src.astype("uint8"), cv2.COLOR_LAB2BGR), 0, 255)
    return result