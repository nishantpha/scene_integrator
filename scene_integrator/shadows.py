import cv2
import numpy as np
from skimage import morphology

def detect_shadows(bg_bgr):
    hsv = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype("float32") / 255.0
    shadow_candidate = (v < 0.4).astype(np.uint8) * 255

    b, g, r = cv2.split(bg_bgr.astype("float32") + 1)
    ratio = (r + g + b) / (3 * np.maximum(r, np.maximum(g, b)))
    color_invariant = (ratio > 0.8).astype(np.uint8) * 255

    shadow_mask = cv2.bitwise_and(shadow_candidate, color_invariant)
    shadow_mask = morphology.remove_small_objects(shadow_mask > 0, min_size=200).astype(np.uint8) * 255

    edges = cv2.Canny(shadow_mask, 50, 150)
    dilated = cv2.dilate(edges, None, iterations=1)
    boundary_strength = cv2.GaussianBlur(dilated, (5, 5), 0)
    hard = cv2.threshold(boundary_strength, 20, 255, cv2.THRESH_BINARY)[1]
    soft = cv2.subtract(shadow_mask, hard)
    return shadow_mask, hard, soft