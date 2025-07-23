import cv2
import numpy as np

def estimate_light_direction_outdoor(bg_bgr, person_point, shadow_point):
    vec = np.array([shadow_point[0]-person_point[0], shadow_point[1]-person_point[1]], dtype=float)
    n = np.linalg.norm(vec) + 1e-6
    return vec / n

def estimate_light_direction_indoor(bg_bgr, samples=2000):
    gray = cv2.cvtColor(bg_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)

    flat_idx = np.argsort(mag.ravel())[-samples:]
    vx = gx.ravel()[flat_idx].mean()
    vy = gy.ravel()[flat_idx].mean()

    vec = np.array([vx, vy], dtype=np.float32)
    n = np.linalg.norm(vec) + 1e-6
    return vec / n