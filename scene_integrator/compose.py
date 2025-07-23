import cv2
import numpy as np
from .segmentation import extract_person_rgba
from .shadows import detect_shadows
from .lighting import estimate_light_direction_indoor, estimate_light_direction_outdoor
from .color_blend import match_color
from .utils import save_debug, ensure_dir
from .config import IntegratorConfig

def integrate_person_into_scene(person_path, bg_path, out_path,
                                config: IntegratorConfig,
                                place_xy=None, scale=1.0,
                                shadow_ref_person=None, shadow_ref_tip=None,
                                debug_dir=None):
    # Load
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)
    person_bgra = extract_person_rgba(person_path)

    # Scale
    if scale != 1.0:
        person_bgra = cv2.resize(person_bgra, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ph, pw = person_bgra.shape[:2]

    # Placement
    if place_xy is None:
        place_xy = (bg.shape[1] // 2 - pw // 2, bg.shape[0] // 2 - ph // 2)
    tx, ty = place_xy

    # Shadow detection (debug only)
    shadow_mask_all, shadow_hard, shadow_soft = detect_shadows(bg)
    if debug_dir and config.debug:
        save_debug(shadow_mask_all, f"{debug_dir}/shadow_all.png", bgr=False)
        save_debug(shadow_hard, f"{debug_dir}/shadow_hard.png", bgr=False)
        save_debug(shadow_soft, f"{debug_dir}/shadow_soft.png", bgr=False)

    # Light direction
    if config.indoor:
        light_vec = estimate_light_direction_indoor(bg, samples=config.light_estimation_samples)
    else:
        if shadow_ref_person is not None and shadow_ref_tip is not None:
            light_vec = estimate_light_direction_outdoor(bg, shadow_ref_person, shadow_ref_tip)
        else:
            light_vec = estimate_light_direction_indoor(bg, samples=config.light_estimation_samples)

    if debug_dir and config.debug:
        dbg = bg.copy()
        p = (tx + pw // 2, ty + ph)
        tip = (int(p[0] + light_vec[0] * 200), int(p[1] + light_vec[1] * 200))
        cv2.arrowedLine(dbg, p, tip, (0, 0, 255), 3, tipLength=0.2)
        save_debug(dbg, f"{debug_dir}/light_vector.png")

    # Alpha & person
    person_bgr = person_bgra[:, :, :3]
    alpha_raw = person_bgra[:, :, 3].astype(np.float32) / 255.0

    # Clean alpha
    hard_mask = (alpha_raw > 0.1).astype(np.float32)
    hard_mask = cv2.morphologyEx(hard_mask, cv2.MORPH_CLOSE,
                                 cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    alpha = cv2.GaussianBlur(hard_mask, (5, 5), 0)
    inside = alpha > 0.6
    alpha[inside] = np.maximum(alpha[inside], config.min_alpha)

    # Color match (optional)
    if config.color_match_method == "none":
        person_color = person_bgr.copy()
    else:
        roi = (max(tx - 30, 0), max(ty - 30, 0),
               min(pw + 60, bg.shape[1] - tx), min(ph + 60, bg.shape[0] - ty))
        ct = match_color(person_bgr, bg, method=config.color_match_method, target_roi=roi)
        person_color = (0.6 * person_bgr + 0.4 * ct).astype(np.uint8)

    # Safe coords
    H, W = bg.shape[:2]
    x0, y0 = tx, ty
    x1, y1 = tx + pw, ty + ph

    bx0 = max(0, x0); by0 = max(0, y0)
    bx1 = min(W, x1); by1 = min(H, y1)

    px0 = bx0 - x0; py0 = by0 - y0
    px1 = px0 + (bx1 - bx0); py1 = py0 + (by1 - by0)

    if bx0 >= bx1 or by0 >= by1:
        raise ValueError("Person placement is outside the background. Adjust --place/--scale.")

    overlay = np.zeros_like(bg)
    overlay[by0:by1, bx0:bx1] = person_color[py0:py1, px0:px1]

    alpha_full = np.zeros((H, W), dtype=np.float32)
    alpha_full[by0:by1, bx0:bx1] = alpha[py0:py1, px0:px1]

    # --- Shrink mask edge (last-resort halo killer) ---
    kernel_edge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # try (3,3) or (5,5)
    alpha_full = cv2.erode(alpha_full, kernel_edge, iterations=1)

    # Base canvas
    canvas = bg.copy()

    # Optional synthetic shadow
    if config.add_shadow:
        if config.contact_shadow_only:
            shadow = synthesize_contact_shadow(alpha_full, strength=config.contact_shadow_strength)
        else:
            shadow = synthesize_full_shadow(alpha_full, light_vec,
                                            strength=config.shadow_opacity,
                                            blur=config.soft_shadow_blur,
                                            contact_strength=config.contact_shadow_strength)
        shadow_bgr = np.dstack([shadow, shadow, shadow])
        canvas = (canvas.astype(float) * (1 - shadow_bgr / 255.0)).astype(np.uint8)

    # -------- Single-pass feathered alpha blend (no Poisson) --------
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (config.edge_inner_erode, config.edge_inner_erode))
    alpha_core = cv2.erode(alpha_full, k, iterations=1)

    ring = np.clip(alpha_full - alpha_core, 0, 1).astype(np.float32)
    if config.use_bilateral:
        ring = cv2.bilateralFilter(ring, config.bilateral_d,
                                   config.bilateral_sigma_color,
                                   config.bilateral_sigma_space)
    else:
        ring = cv2.GaussianBlur(ring, (config.edge_ring_blur, config.edge_ring_blur), 0)
    ring = np.clip(ring, 0.0, 1.0)
    ring[ring < 0.07] = 0

    alpha_feather = np.clip(alpha_core + ring, 0, 1)

    # sample solid FG color (median of core area)
    core = overlay[alpha_core > 0.95].reshape(-1, 3)
    fg_ref = np.median(core, axis=0) if core.size else np.array([0,0,0])
    fg_ref_img = np.tile(fg_ref, (overlay.shape[0], overlay.shape[1], 1))

    final_overlay = overlay.astype(float) * (1 - ring[:, :, None]) + fg_ref_img * ring[:, :, None]

    canvas = (final_overlay * alpha_feather[:, :, None] +
              canvas.astype(float) * (1 - alpha_feather[:, :, None])).astype(np.uint8)
    # ---------------------------------------------------------------

    if debug_dir and config.debug:
        ensure_dir(debug_dir)
        save_debug(overlay, f"{debug_dir}/person_color.png")
        save_debug((alpha_full * 255).astype(np.uint8), f"{debug_dir}/alpha.png", bgr=False)
        save_debug((alpha_feather * 255).astype(np.uint8), f"{debug_dir}/alpha_feather.png", bgr=False)

    cv2.imwrite(out_path, canvas)
    return canvas

# ---------------------------------------------------------------------------

def synthesize_full_shadow(alpha_full, light_vec, strength=0.6, blur=21, contact_strength=0.4):
    h, w = alpha_full.shape
    shadow = np.zeros((h, w), dtype=float)
    steps = 80
    for i in range(1, steps):
        dx = int(light_vec[0] * i)
        dy = int(light_vec[1] * i)
        shifted = np.roll(alpha_full, shift=(dy, dx), axis=(0, 1))
        decay = np.exp(-i / steps)
        shadow = np.maximum(shadow, shifted * decay)
    shadow = cv2.GaussianBlur(shadow, (blur, blur), 0)
    shadow *= 255 * strength
    contact = cv2.GaussianBlur(alpha_full, (5, 5), 0) * 255 * contact_strength
    shadow = np.maximum(shadow, contact)
    return shadow.astype(np.uint8)

def synthesize_contact_shadow(alpha_full, strength=0.4):
    contact = cv2.GaussianBlur(alpha_full, (11, 11), 0) * 255 * strength
    return contact.astype(np.uint8)