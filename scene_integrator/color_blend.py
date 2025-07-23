import numpy as np
import cv2
from .utils import reinhard_color_transfer

def match_color(person_bgr, bg_bgr, method="reinhard", target_roi=None):
    if target_roi:
        x, y, w, h = target_roi
        tgt = bg_bgr[y:y+h, x:x+w]
    else:
        tgt = bg_bgr

    if method == "reinhard":
        return reinhard_color_transfer(person_bgr, tgt)
    elif method == "hist":
        return hist_match(person_bgr, tgt)
    elif method == "luma":
        return luma_match(person_bgr, tgt)
    else:
        return person_bgr

def hist_match(src, template):
    matched = np.zeros_like(src)
    for i in range(3):
        s = src[:, :, i].ravel()
        t = template[:, :, i].ravel()
        s_values, bin_idx, s_counts = np.unique(s, return_inverse=True, return_counts=True)
        t_values, t_counts = np.unique(t, return_counts=True)
        s_quantiles = np.cumsum(s_counts).astype(np.float64) / s.size
        t_quantiles = np.cumsum(t_counts).astype(np.float64) / t.size
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
        matched[:, :, i] = interp_t_values[bin_idx].reshape(src[:, :, i].shape)
    return matched.astype(src.dtype)

def luma_match(src_bgr, tgt_bgr):
    """
    Match mean/var of luminance (Y channel) only, keep chroma.
    """
    src_ycrcb = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    tgt_ycrcb = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)

    sY = src_ycrcb[:, :, 0]
    tY = tgt_ycrcb[:, :, 0]

    sm, ss = sY.mean(), sY.std() + 1e-6
    tm, ts = tY.mean(), tY.std() + 1e-6

    sY_new = (sY - sm) * (ts / ss) + tm
    src_ycrcb[:, :, 0] = np.clip(sY_new, 0, 255)

    return cv2.cvtColor(src_ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)