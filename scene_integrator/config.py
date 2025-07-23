from dataclasses import dataclass

@dataclass
class IntegratorConfig:
    # lighting / scene
    indoor: bool = False
    person_height_m: float = 1.72
    light_estimation_samples: int = 2000

    # blending
    poisson_blend: bool = False
    color_match_method: str = "none"  # "none", "reinhard", "hist"
    debug: bool = True

    # shadows
    add_shadow: bool = False
    contact_shadow_only: bool = True
    shadow_opacity: float = 0.45
    soft_shadow_blur: int = 21
    contact_shadow_strength: float = 0.4

    # alpha cleanup
    min_alpha: float = 0.97

    # edge feather tuning
    edge_inner_erode: int = 1
    edge_ring_blur: int = 7
    bg_bleed: float = 0.12

    # bilateral alternative
    use_bilateral: bool = False
    bilateral_d: int = 9
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 25.0

    # ---- NEW: FBA / IndexNet matting ----
    use_fba: bool = True
    fba_weights_path: str = "models/fba_matting.pth"