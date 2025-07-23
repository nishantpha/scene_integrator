from dataclasses import dataclass

@dataclass
class IntegratorConfig:
    # lighting / scene
    indoor: bool = False
    person_height_m: float = 1.72
    light_estimation_samples: int = 2000

    # blending
    poisson_blend: bool = False          # keep off unless really needed
    color_match_method: str = "none"     # "none", "reinhard", "hist"
    debug: bool = True

    # shadows
    add_shadow: bool = False
    contact_shadow_only: bool = True
    shadow_opacity: float = 0.45
    soft_shadow_blur: int = 21
    contact_shadow_strength: float = 0.4

    # alpha cleanup
    min_alpha: float = 0.85

    # edge feather tuning
    edge_inner_erode: int = 5            # odd kernel
    edge_ring_blur: int = 13             # kept for fallback (Gaussian)
    bg_bleed: float = 0.35               # 0..1 background fraction in feather mix

    # bilateral filter (edge-preserving) instead of Gaussian on feather ring
    use_bilateral: bool = True
    bilateral_d: int = 9                 # diameter of each pixel neighborhood
    bilateral_sigma_color: float = 25.0  # color sigma
    bilateral_sigma_space: float = 25.0  # space sigma