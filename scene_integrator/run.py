import argparse
from pathlib import Path
from .compose import integrate_person_into_scene
from .config import IntegratorConfig
from .utils import ensure_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--person', required=True)
    parser.add_argument('--background', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--debug-dir', default=None)

    parser.add_argument('--indoor', action='store_true')
    parser.add_argument('--person-height', type=float, default=1.72)

    parser.add_argument('--place-x', type=int, default=None)
    parser.add_argument('--place-y', type=int, default=None)

    parser.add_argument('--scale', type=float, default=1.0,
                        help='Manual scale factor for the person image.')
    parser.add_argument('--fit-height', type=float, default=None,
                        help='Auto-scale: person height = this fraction of background height (0–1). Overrides --scale.')

    parser.add_argument('--add-shadow', action='store_true')
    parser.add_argument('--full-shadow', action='store_true')

    parser.add_argument('--color-match', choices=['none', 'reinhard', 'hist'], default='none')

    args = parser.parse_args()

    ensure_dir(Path(args.out).parent)
    if args.debug_dir:
        ensure_dir(args.debug_dir)

    cfg = IntegratorConfig(
        indoor=args.indoor,
        person_height_m=args.person_height,
        color_match_method=args.color_match,
        debug=args.debug_dir is not None,
        add_shadow=args.add_shadow,
        contact_shadow_only=not args.full_shadow
    )

    place = None
    if args.place_x is not None and args.place_y is not None:
        place = (args.place_x, args.place_y)

    integrate_person_into_scene(
        person_path=args.person,
        bg_path=args.background,
        out_path=args.out,
        config=cfg,
        place_xy=place,
        scale=args.scale,
        fit_height=args.fit_height,  # <---- NEW
        debug_dir=args.debug_dir
    )

    print(f"✅ Done. Wrote {args.out}")


if __name__ == '__main__':
    main()