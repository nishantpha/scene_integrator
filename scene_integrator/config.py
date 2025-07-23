from dataclasses import dataclass

@dataclass
class IntegratorConfig:
    # Scene / lighting (kept for API completeness)
    indoor: bool = False
    person_height_m: float = 1.72
    light_estimation_samples: int = 2000

    # Blending
    poisson_blend: bool = False          # Always False in this rollback
    color_match_method: str = "none"     # "none", "reinhard", "hist"
    debug: bool = True

    # Shadows
    add_shadow: bool = False
    contact_shadow_only: bool = True
    shadow_opacity: float = 0.45
    soft_shadow_blur: int = 21
    contact_shadow_strength: float = 0.4

    # Alpha cleanup
    min_alpha: float = 0.98

    # Edge feather tuning
    edge_inner_erode: int = 1  # odd kernel
    edge_ring_blur: int = 7  # kept for fallback (Gaussian)
    bg_bleed: float = 0.12 # 0..1 background fraction in feather mix

    # Bilateral switch (off by default; Gaussian is fine)
    use_bilateral: bool = False
    bilateral_d: int = 9 # diameter of each pixel neighborhood
    bilateral_sigma_color: float = 25.0 # color sigma
    bilateral_sigma_space: float = 25.0 # space sigma

        # --- lighting match ---
    color_match_method: str = "none"  # keep, but we'll add "luma"
    apply_bg_illumination: bool = True
    illum_gauss: int = 61            # odd, large → very soft light map
    illum_strength: float = 0.35     # 0..1 how strongly to multiply person by bg light map
    gamma_after: float = 1.0         # final gamma tweak on person (1=no change)