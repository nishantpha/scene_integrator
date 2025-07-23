from dataclasses import dataclass

@dataclass
class IntegratorConfig:
    # lighting / scene
    indoor: bool = False
    person_height_m: float = 1.72
    light_estimation_samples: int = 2000

    # blending
    poisson_blend: bool = False          # keep off unless you really need it
    color_match_method: str = "none"     # "none", "reinhard", or "hist"
    debug: bool = True

    # shadows
    add_shadow: bool = False             # no synthetic shadow by default
    contact_shadow_only: bool = True
    shadow_opacity: float = 0.45
    soft_shadow_blur: int = 21
    contact_shadow_strength: float = 0.4

    # alpha cleanup
    min_alpha: float = 0.85              # clamp inside person to avoid transparency

    # edge feather tuning
    edge_inner_erode: int = 5      # shrink solid mask by this many px (odd kernel)
    edge_ring_blur: int = 17      # blur size for the feather ring (odd kernel)
    bg_bleed: float = 0.35         # 0..1 how much background to mix into the edge