Scene Integrator (CPU‑Only, Mac-Friendly)

Photorealistically place a person cutout into a new background using pure Python/OpenCV plus rembg (U²-Net ONNX). This branch purposefully avoids Poisson blending and heavy matting nets because they caused ghosting on CPU. We rely on a robust single‑pass feathered alpha blend with optional color transfer and (optional) synthetic contact shadows.

⸻

✨ Features
• Background removal via rembg + U²‑Net (ONNX).
• Automatic or manual placement & scaling of the person.
• Optional light/shadow estimation (simple indoor heuristic).
• Single‑pass edge feathering (no double exposure, no transparency bugs).
• Optional color transfer (Reinhard or histogram matching).
• Runs entirely on CPU (tested on macOS M1 Air, Python 3.11).

⸻

1. Requirements
   • Python 3.11.x (OpenCV wheels for 3.13 are still incomplete).
   • macOS / Linux / Windows (no GPU needed).
   • Command line (bash/zsh/CMD).

⸻

2. Project Structure

scene_integrator_project/
├── README.md
├── requirements.txt
├── .gitignore
├── background.jpeg
├── person.jpeg
├── models/
│ └── u2net.onnx
├── outputs/
├── scene_integrator/
│ ├── **init**.py
│ ├── config.py
│ ├── utils.py
│ ├── segmentation.py
│ ├── shadows.py
│ ├── lighting.py
│ ├── color_blend.py
│ ├── compose.py
│ └── run.py
└── venv/ (after you create it)

models/ and outputs/ are created by you; venv/ appears after you set up the virtual environment.

⸻

3. Setup

# Clone your repo

git clone <your-repo-url> scene_integrator_project
cd scene_integrator_project

# Create & activate a Python 3.11 virtual environment

python3.11 -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

# Install dependencies

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

Download the U²‑Net model (once)

mkdir -p models
curl -L -o models/u2net.onnx \
 https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx

If curl is missing: brew install curl (macOS) or use a browser and save into models/.

⸻

4. Put Your Images
   • person.jpeg – front-view portrait to cut out.
   • background.jpeg – destination scene (outdoor/indoor).
   Place them in the repo root or pass absolute paths when running.

⸻

5. Run the Pipeline

mkdir -p outputs outputs/debug

python -m scene_integrator.run \
 --person person.jpeg \
 --background background.jpeg \
 --out outputs/composite.jpg \
 --debug-dir outputs/debug \
 --color-match none \
 --scale 1.0

Useful flags

Flag Meaning Default
--scale 0.5 Resize person by factor 1.0
--place-x 300 --place-y 200 Manually set top-left placement auto-center
--color-match reinhard Color transfer (“none”, “reinhard”, “hist”) none
--add-shadow Add synthetic shadow Off
--full-shadow If --add-shadow, make a long cast shadow Contact only
--indoor Use indoor light heuristic for direction Outdoor fallback

Outputs:
• outputs/composite.jpg – final result
• outputs/debug/alpha.png, alpha*feather.png, person_color.png, shadow*\*.png – diagnostics

⸻

6. Common Issues & Quick Fixes

Symptom Cause Fix
Empty outputs/ Folder missing / exception suppressed Ensure outputs/ exists, check console traceback
ModuleNotFoundError: cv2 etc. Deps not installed in venv python -m pip install -r requirements.txt
Person looks pasted / halo edge Feather band too wide or too much BG bleed Lower edge_inner_erode, edge_ring_blur, bg_bleed; raise min_alpha
Skin/clothes color off Aggressive color transfer Use --color-match none
Double/transparent person Poisson clone stacking Poisson is disabled; use this branch

⸻

7. Tuning Parameters (scene_integrator/config.py)

min_alpha = 0.95 # Clamp interior opacity
edge_inner_erode = 3 # Shrink solid mask radius (odd number)
edge_ring_blur = 11 # Gaussian blur size for feather ring (odd)
bg_bleed = 0.25 # BG mix ratio in ring (0..1)
use_bilateral = False # True = edge‑preserving but slower

How they affect halos:
• ↓ edge_inner_erode → thinner feather ring, less gray fog.
• ↓ edge_ring_blur → sharper edge.
• ↓ bg_bleed → less background contamination (less gray rim).
• ↑ min_alpha → interior less transparent.

⸻

8. Extending the Project
   • Swap rembg model for a different ONNX (u2netp, isnet, etc.).
   • Add Poisson blending back behind a flag (ensure single composite path!).
   • Integrate better matting nets (MODNet/FBA) if you can download large weights.
   • Auto-scale person using perspective cues.

⸻

9. Git Tips

Add .gitignore (provided) before first commit so outputs/ and models/ don’t clog your repo. If you want to version models/u2net.onnx, use Git LFS.

git lfs install
git lfs track "models/\*.onnx"

⸻

10. License

MIT (or your preferred license). Update this section as needed.

⸻

Happy compositing! 🎨
