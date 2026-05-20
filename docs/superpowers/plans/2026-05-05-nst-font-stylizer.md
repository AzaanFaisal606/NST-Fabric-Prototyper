# NST Font Stylizer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local web app that stylizes font glyphs using Neural Style Transfer (Gatys et al. 2016), with a React frontend, FastAPI backend, and a DIP pre/post-processing pipeline justifying classical-DIP coverage for an academic course.

**Architecture:** Single-page React (Vite + Tailwind) talks to a FastAPI server. Backend renders glyphs with PIL, runs DIP preprocess → frozen avg-pool VGG-19 feature extraction → L-BFGS NST optimization → DIP postprocess, and returns PNG glyphs via static files. Jobs run in `BackgroundTasks` with in-memory state and on-disk artifacts.

**Tech Stack:** Python 3.11, FastAPI, PyTorch, torchvision, PIL, scikit-image, scipy, OpenCV, React 18, Vite, Tailwind CSS.

**Spec reference:** `docs/superpowers/specs/2026-05-05-nst-font-stylizer-design.md`

**Testing approach:** No formal test suite (per spec §9). Each task uses **manual verification steps** (run a command / hit an endpoint / eyeball output). Frequent commits.

**Code style mandate (per user, applies to ALL backend tasks):**
- Concise. No overengineering, no premature abstraction.
- One short comment line before each major block of NST/DIP code, explaining what the block does.
- Inline labels for important variables (e.g. `# content layer`, `# style layers`, `# Gram matrix`).
- Well-formatted, easily readable, easy to explain in a course defense.

---

## File Structure

```
project/
├── backend/
│   ├── main.py
│   ├── nst.py
│   ├── vgg.py
│   ├── glyph_renderer.py
│   ├── preprocessing.py
│   ├── postprocessing.py
│   ├── jobs/                        # runtime artifacts (gitignored)
│   ├── fonts/                       # bundled .ttf
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   ├── postcss.config.js
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── index.css
│       ├── api.js
│       └── components/
│           ├── StyleUploader.jsx
│           ├── FontSelector.jsx
│           ├── CharSetSelector.jsx
│           ├── ParamSliders.jsx
│           ├── StylizeButton.jsx
│           ├── ProgressBar.jsx
│           └── ResultGrid.jsx
├── .gitignore
└── README.md
```

**File responsibilities:**
- `glyph_renderer.py` — PIL glyph rasterization
- `preprocessing.py` — DIP preprocessing (style + content)
- `vgg.py` — VGG-19 loader (avg-pool swap) + feature extractor
- `nst.py` — Gram matrix + L-BFGS optimization loop
- `postprocessing.py` — DIP postprocessing (median, Butterworth, hist match)
- `main.py` — FastAPI app, endpoints, job state, background task

---

## Task 1: Repo skeleton + .gitignore + README stub

**Files:**
- Create: `.gitignore`
- Create: `README.md`
- Create: `backend/`, `backend/fonts/`, `backend/jobs/.keep`
- Create: `frontend/`

- [ ] **Step 1.1: Create top-level dirs**

```bash
mkdir -p backend/fonts backend/jobs frontend
touch backend/jobs/.keep
```

- [ ] **Step 1.2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
.venv/
venv/

# Node
node_modules/
dist/
.vite/

# Runtime artifacts
backend/jobs/*
!backend/jobs/.keep

# OS
.DS_Store
Thumbs.db

# IDE
.vscode/
.idea/
```

- [ ] **Step 1.3: Write `README.md` stub**

```markdown
# NST Font Stylizer

Local web app that stylizes font glyphs using Neural Style Transfer.

## Run backend
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

## Run frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.
```

- [ ] **Step 1.4: Verify**

Run: `ls -la`
Expected: `.gitignore`, `README.md`, `backend/`, `frontend/` all present.

- [ ] **Step 1.5: Commit**

```bash
git init
git add .gitignore README.md backend frontend
git commit -m "chore: initial repo skeleton"
```

---

## Task 2: Backend Python environment + requirements

**Files:**
- Create: `backend/requirements.txt`

- [ ] **Step 2.1: Write `backend/requirements.txt`**

```
fastapi==0.115.0
uvicorn[standard]==0.30.6
python-multipart==0.0.10
torch==2.4.1
torchvision==0.19.1
pillow==10.4.0
numpy==1.26.4
scipy==1.13.1
scikit-image==0.24.0
opencv-python-headless==4.10.0.84
```

- [ ] **Step 2.2: Create venv + install**

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

- [ ] **Step 2.3: Verify torch + cuda detection**

Run:
```bash
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
```
Expected: prints torch version and `cuda True` or `cuda False` without error.

- [ ] **Step 2.4: Commit**

```bash
git add backend/requirements.txt
git commit -m "chore: add backend requirements"
```

---

## Task 3: Bundle fonts

**Files:**
- Add: `backend/fonts/Merriweather-Regular.ttf`
- Add: `backend/fonts/Roboto-Regular.ttf`
- Add: `backend/fonts/JetBrainsMono-Regular.ttf`
- Add: `backend/fonts/PlayfairDisplay-Regular.ttf`
- Add: `backend/fonts/BebasNeue-Regular.ttf`
- Add: `backend/fonts/Lobster-Regular.ttf`

- [ ] **Step 3.1: Download fonts from Google Fonts**

Run (one-time, manual):
```bash
cd backend/fonts
# Merriweather
curl -L -o Merriweather.zip "https://fonts.google.com/download?family=Merriweather"
unzip -j Merriweather.zip "static/Merriweather-Regular.ttf" -d .
rm Merriweather.zip

# Roboto
curl -L -o Roboto.zip "https://fonts.google.com/download?family=Roboto"
unzip -j Roboto.zip "static/Roboto-Regular.ttf" -d .
rm Roboto.zip

# JetBrains Mono
curl -L -o JetBrainsMono.zip "https://fonts.google.com/download?family=JetBrains+Mono"
unzip -j JetBrainsMono.zip "static/JetBrainsMono-Regular.ttf" -d .
rm JetBrainsMono.zip

# Playfair Display
curl -L -o PlayfairDisplay.zip "https://fonts.google.com/download?family=Playfair+Display"
unzip -j PlayfairDisplay.zip "static/PlayfairDisplay-Regular.ttf" -d .
rm PlayfairDisplay.zip

# Bebas Neue
curl -L -o BebasNeue.zip "https://fonts.google.com/download?family=Bebas+Neue"
unzip -j BebasNeue.zip "BebasNeue-Regular.ttf" -d .
rm BebasNeue.zip

# Lobster
curl -L -o Lobster.zip "https://fonts.google.com/download?family=Lobster"
unzip -j Lobster.zip "Lobster-Regular.ttf" -d .
rm Lobster.zip
```

If Google Fonts download URLs change, fetch manually from https://fonts.google.com and place `*-Regular.ttf` files into `backend/fonts/`.

- [ ] **Step 3.2: Verify**

Run: `ls backend/fonts/*.ttf | wc -l`
Expected: `6`

- [ ] **Step 3.3: Commit**

```bash
git add backend/fonts/*.ttf
git commit -m "chore: bundle 6 google fonts"
```

---

## Task 4: Glyph renderer

**Files:**
- Create: `backend/glyph_renderer.py`

- [ ] **Step 4.1: Write `backend/glyph_renderer.py`**

```python
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# directory holding bundled .ttf files
FONTS_DIR = Path(__file__).parent / "fonts"

# map of font key -> filename
FONTS = {
    "Merriweather":      "Merriweather-Regular.ttf",
    "Roboto":            "Roboto-Regular.ttf",
    "JetBrainsMono":     "JetBrainsMono-Regular.ttf",
    "PlayfairDisplay":   "PlayfairDisplay-Regular.ttf",
    "BebasNeue":         "BebasNeue-Regular.ttf",
    "Lobster":           "Lobster-Regular.ttf",
}


def list_fonts() -> list[str]:
    # return font keys for available bundled fonts
    return [k for k, fname in FONTS.items() if (FONTS_DIR / fname).exists()]


def render_glyph(char: str, font_key: str, size: int = 512, padding: int = 40) -> Image.Image:
    # white canvas, black glyph centered, auto-fit font size
    if font_key not in FONTS:
        raise ValueError(f"unknown font: {font_key}")
    font_path = FONTS_DIR / FONTS[font_key]

    canvas = Image.new("RGB", (size, size), color="white")
    draw = ImageDraw.Draw(canvas)

    # binary search font size so glyph fits within (size - 2*padding)
    target = size - 2 * padding
    lo, hi = 10, 800
    best_font = ImageFont.truetype(str(font_path), 10)
    while lo <= hi:
        mid = (lo + hi) // 2
        f = ImageFont.truetype(str(font_path), mid)
        bbox = draw.textbbox((0, 0), char, font=f)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w <= target and h <= target:
            best_font = f
            lo = mid + 1
        else:
            hi = mid - 1

    # center using bbox offset
    bbox = draw.textbbox((0, 0), char, font=best_font)
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - w) / 2 - bbox[0]
    y = (size - h) / 2 - bbox[1]
    draw.text((x, y), char, fill="black", font=best_font)
    return canvas
```

- [ ] **Step 4.2: Manual verify**

Run from `backend/`:
```bash
python -c "from glyph_renderer import render_glyph; render_glyph('A', 'Roboto').save('/tmp/test_glyph_A.png')"
```
Open `/tmp/test_glyph_A.png` — expect black 'A' centered on white 512x512 background, large but with visible padding.

Repeat for one more font:
```bash
python -c "from glyph_renderer import render_glyph; render_glyph('Q', 'Lobster').save('/tmp/test_glyph_Q.png')"
```
Eyeball both PNGs. Both should look clean.

- [ ] **Step 4.3: Commit**

```bash
git add backend/glyph_renderer.py
git commit -m "feat: glyph renderer with auto-fit font sizing"
```

---

## Task 5: VGG-19 feature extractor

**Files:**
- Create: `backend/vgg.py`

- [ ] **Step 5.1: Write `backend/vgg.py`**

```python
import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# VGG-19 module index map for the layers we need (paper §3)
LAYER_INDEX = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,    # content layer
    "conv5_1": 28,
}

# style layers (multi-scale texture, paper §3)
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
# content layer (paper §3.2)
CONTENT_LAYER = "conv4_2"


def load_vgg(device: torch.device) -> nn.Sequential:
    # load pretrained VGG-19 features, swap max-pool for avg-pool (paper §2), freeze
    features = vgg19(weights=VGG19_Weights.DEFAULT).features
    swapped = nn.Sequential()
    for i, layer in enumerate(features):
        if isinstance(layer, nn.MaxPool2d):
            swapped.add_module(str(i), nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            swapped.add_module(str(i), layer)
    swapped.eval().to(device)
    for p in swapped.parameters():
        p.requires_grad_(False)
    return swapped


def extract_features(x: torch.Tensor, vgg: nn.Sequential, layer_names: list[str]) -> dict[str, torch.Tensor]:
    # forward pass, capture activations at requested named layers
    wanted = {LAYER_INDEX[n]: n for n in layer_names}
    feats: dict[str, torch.Tensor] = {}
    h = x
    last = max(wanted.keys())
    for i, layer in enumerate(vgg):
        h = layer(h)
        if i in wanted:
            feats[wanted[i]] = h
        if i >= last:
            break
    return feats
```

- [ ] **Step 5.2: Manual verify**

Run from `backend/`:
```bash
python -c "
import torch
from vgg import load_vgg, extract_features, STYLE_LAYERS, CONTENT_LAYER
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = load_vgg(device)
x = torch.randn(1, 3, 512, 512, device=device)
feats = extract_features(x, vgg, STYLE_LAYERS + [CONTENT_LAYER])
for k, v in feats.items(): print(k, tuple(v.shape))
print('avg-pool count:', sum(1 for m in vgg if isinstance(m, torch.nn.AvgPool2d)))
print('max-pool count:', sum(1 for m in vgg if isinstance(m, torch.nn.MaxPool2d)))
"
```
Expected: prints 6 layer names with descending spatial dims (512→256→128→64→32→32). `avg-pool count: 5`, `max-pool count: 0`. No errors.

- [ ] **Step 5.3: Commit**

```bash
git add backend/vgg.py
git commit -m "feat: vgg-19 loader with avg-pool swap and named feature extraction"
```

---

## Task 6: DIP preprocessing module

**Files:**
- Create: `backend/preprocessing.py`

- [ ] **Step 6.1: Write `backend/preprocessing.py`**

```python
import numpy as np
import torch
import cv2
from PIL import Image
from torchvision import transforms

# ImageNet normalization (VGG was trained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SIZE = 512


def _to_tensor(img: Image.Image, device: torch.device) -> torch.Tensor:
    # PIL -> normalized [1,3,H,W] tensor on device
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return t(img).unsqueeze(0).to(device)


def _resize_center_crop(img: Image.Image, size: int) -> Image.Image:
    # resize so short side = size, then center crop to size x size
    w, h = img.size
    scale = size / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    left = (new_w - size) // 2
    top  = (new_h - size) // 2
    return img.crop((left, top, left + size, top + size))


def preprocess_style(img: Image.Image, device: torch.device) -> torch.Tensor:
    # resize+crop -> RGB hist eq (per channel) -> Gaussian blur -> normalize
    img = img.convert("RGB")
    img = _resize_center_crop(img, TARGET_SIZE)
    arr = np.array(img)

    # per-channel histogram equalization
    eq = np.zeros_like(arr)
    for c in range(3):
        eq[..., c] = cv2.equalizeHist(arr[..., c])

    # light Gaussian blur (sigma=0.5) to suppress sensor noise
    blurred = cv2.GaussianBlur(eq, ksize=(0, 0), sigmaX=0.5, sigmaY=0.5)

    out = Image.fromarray(blurred)
    return _to_tensor(out, device)


def preprocess_content(glyph: Image.Image, device: torch.device, k: float = 0.5) -> torch.Tensor:
    # Laplacian sharpening: out = glyph - k * laplacian(glyph)
    arr = np.array(glyph.convert("RGB")).astype(np.float32)
    lap = cv2.Laplacian(arr, ddepth=cv2.CV_32F, ksize=3)
    sharp = np.clip(arr - k * lap, 0, 255).astype(np.uint8)
    return _to_tensor(Image.fromarray(sharp), device)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    # invert ImageNet normalization, clamp to [0,1] for display/save
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)
```

- [ ] **Step 6.2: Manual verify**

Place any test JPG at `/tmp/test_style.jpg` (or use any local image). From `backend/`:
```bash
python -c "
import torch
from PIL import Image
from preprocessing import preprocess_style, preprocess_content, denormalize
from glyph_renderer import render_glyph
device = torch.device('cpu')
style = Image.open('/tmp/test_style.jpg')
st = preprocess_style(style, device)
print('style tensor:', tuple(st.shape), st.min().item(), st.max().item())
glyph = render_glyph('A', 'Roboto')
ct = preprocess_content(glyph, device)
print('content tensor:', tuple(ct.shape))
# round-trip eyeball check
from torchvision.transforms.functional import to_pil_image
to_pil_image(denormalize(st).squeeze(0).cpu()).save('/tmp/preproc_style.png')
to_pil_image(denormalize(ct).squeeze(0).cpu()).save('/tmp/preproc_content.png')
"
```
Expected: prints `style tensor: (1, 3, 512, 512)` and `content tensor: (1, 3, 512, 512)` without error. Eyeball `/tmp/preproc_style.png` (style with boosted contrast, slight blur) and `/tmp/preproc_content.png` (sharper-edged 'A' glyph).

- [ ] **Step 6.3: Commit**

```bash
git add backend/preprocessing.py
git commit -m "feat: dip preprocessing - hist eq, gaussian blur, laplacian sharpening"
```

---

## Task 7: NST core (Gram + L-BFGS loop)

**Files:**
- Create: `backend/nst.py`

- [ ] **Step 7.1: Write `backend/nst.py`**

```python
from typing import Callable, Optional
import torch
import torch.nn.functional as F
from vgg import extract_features, STYLE_LAYERS, CONTENT_LAYER


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    # feature correlations across channels, normalized (paper eq. 3 / our normalization)
    b, c, h, w = feat.shape
    flat = feat.view(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


def stylize(
    content_t: torch.Tensor,
    style_t: torch.Tensor,
    vgg: torch.nn.Sequential,
    alpha: float,
    beta: float,
    iters: int,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> torch.Tensor:
    # content features (target for L_content)
    content_target = extract_features(content_t, vgg, [CONTENT_LAYER])[CONTENT_LAYER].detach()

    # style Gram matrices (targets for L_style)
    style_feats = extract_features(style_t, vgg, STYLE_LAYERS)
    style_targets = {l: gram_matrix(style_feats[l]).detach() for l in STYLE_LAYERS}

    # init G as content image (paper Fig 6 - deterministic, biased to letter shape)
    G = content_t.clone().requires_grad_(True)

    # L-BFGS optimizer on pixels of G (paper §3)
    optimizer = torch.optim.LBFGS([G], lr=1.0, max_iter=1)

    # equal weights across the 5 style layers (paper §3)
    w_l = 1.0 / len(STYLE_LAYERS)

    step = {"i": 0}

    def closure():
        # closure runs each L-BFGS iteration
        optimizer.zero_grad()

        # extract G features at all layers we need
        feats = extract_features(G, vgg, STYLE_LAYERS + [CONTENT_LAYER])

        # content loss (MSE at conv4_2)
        content_loss = F.mse_loss(feats[CONTENT_LAYER], content_target)

        # style loss = sum_l w_l * MSE(gram(G), gram(style)) at each style layer
        style_loss = torch.zeros((), device=G.device)
        for l in STYLE_LAYERS:
            style_loss = style_loss + w_l * F.mse_loss(gram_matrix(feats[l]), style_targets[l])

        # total loss = α * content + β * style
        loss = alpha * content_loss + beta * style_loss
        loss.backward()
        return loss

    # main optimization loop
    for i in range(iters):
        optimizer.step(closure)
        step["i"] = i + 1
        if progress_cb is not None:
            progress_cb(step["i"])

    return G.detach()
```

- [ ] **Step 7.2: Manual verify (50-iter smoke run)**

From `backend/`:
```bash
python -c "
import torch
from PIL import Image
from glyph_renderer import render_glyph
from preprocessing import preprocess_style, preprocess_content, denormalize
from vgg import load_vgg
from nst import stylize
from torchvision.transforms.functional import to_pil_image
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vgg = load_vgg(device)
glyph = render_glyph('A', 'Roboto')
ct = preprocess_content(glyph, device)
st = preprocess_style(Image.open('/tmp/test_style.jpg'), device)
def cb(i):
    if i % 10 == 0: print('iter', i)
G = stylize(ct, st, vgg, alpha=1.0, beta=1e4, iters=50, progress_cb=cb)
to_pil_image(denormalize(G).squeeze(0).cpu()).save('/tmp/nst_out_A.png')
print('saved /tmp/nst_out_A.png')
"
```
Expected: prints `iter 10, 20, 30, 40, 50`, saves PNG. Eyeball `/tmp/nst_out_A.png` — should look like 'A' with style-image colors/textures applied.

- [ ] **Step 7.3: Commit**

```bash
git add backend/nst.py
git commit -m "feat: nst core - gram matrix and L-BFGS optimization loop"
```

---

## Task 8: DIP postprocessing module

**Files:**
- Create: `backend/postprocessing.py`

- [ ] **Step 8.1: Write `backend/postprocessing.py`**

```python
import numpy as np
from PIL import Image, ImageFilter
from skimage.exposure import match_histograms


def _butterworth_lowpass(arr: np.ndarray, cutoff: float = 0.3, order: int = 2) -> np.ndarray:
    # frequency-domain low-pass via FFT (Butterworth filter, per channel)
    h, w = arr.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    # normalized radial distance from center, in [0, ~sqrt(2)/2]
    d = np.sqrt(((y - cy) / h) ** 2 + ((x - cx) / w) ** 2)
    # Butterworth transfer function H(u,v) = 1 / (1 + (D/D0)^(2n))
    H = 1.0 / (1.0 + (d / cutoff) ** (2 * order))

    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        F = np.fft.fftshift(np.fft.fft2(arr[..., c].astype(np.float32)))
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return np.clip(out, 0, 255).astype(np.uint8)


def postprocess(stylized_pil: Image.Image, style_pil: Image.Image) -> Image.Image:
    # median filter (3x3) - removes salt-and-pepper noise from NST optimization
    img = stylized_pil.filter(ImageFilter.MedianFilter(size=3))

    # Butterworth low-pass (frequency domain) - smooths high-freq artifacts
    arr = np.array(img.convert("RGB"))
    arr = _butterworth_lowpass(arr, cutoff=0.3, order=2)

    # histogram matching to style image - re-aligns color distribution to style palette
    style_arr = np.array(style_pil.convert("RGB").resize(img.size, Image.LANCZOS))
    matched = match_histograms(arr, style_arr, channel_axis=-1).astype(np.uint8)

    return Image.fromarray(matched)
```

- [ ] **Step 8.2: Manual verify**

From `backend/`:
```bash
python -c "
from PIL import Image
from postprocessing import postprocess
stylized = Image.open('/tmp/nst_out_A.png')
style = Image.open('/tmp/test_style.jpg')
out = postprocess(stylized, style)
out.save('/tmp/post_out_A.png')
print('saved /tmp/post_out_A.png  size:', out.size)
"
```
Expected: prints `saved ... size: (512, 512)`. Eyeball `/tmp/post_out_A.png` vs `/tmp/nst_out_A.png` — postprocess output should look smoother and have color palette closer to style image.

- [ ] **Step 8.3: Commit**

```bash
git add backend/postprocessing.py
git commit -m "feat: dip postprocessing - median, butterworth lowpass, hist match"
```

---

## Task 9: FastAPI app — startup, fonts endpoint, CORS

**Files:**
- Create: `backend/main.py`

- [ ] **Step 9.1: Write initial `backend/main.py`**

```python
import logging
from pathlib import Path
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from glyph_renderer import list_fonts
from vgg import load_vgg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nst")

JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

app = FastAPI(title="NST Font Stylizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# device + VGG loaded once at startup, kept in memory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"device: {device}")
vgg = load_vgg(device)
log.info("vgg-19 loaded (avg-pool swap, frozen)")


@app.get("/fonts")
def get_fonts():
    return {"fonts": list_fonts()}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}
```

- [ ] **Step 9.2: Manual verify**

From `backend/`:
```bash
uvicorn main:app --reload
```
In another terminal:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/fonts
```
Expected:
- `/health` → `{"status":"ok","device":"cuda"}` or `cpu`
- `/fonts` → `{"fonts":["Merriweather","Roboto","JetBrainsMono","PlayfairDisplay","BebasNeue","Lobster"]}`

Stop the server (Ctrl+C).

- [ ] **Step 9.3: Commit**

```bash
git add backend/main.py
git commit -m "feat: fastapi app skeleton - startup, fonts endpoint, cors"
```

---

## Task 10: Job state + /stylize endpoint

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 10.1: Replace `backend/main.py` with full version including `/stylize`**

```python
import io
import logging
import uuid
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from glyph_renderer import list_fonts, render_glyph
from preprocessing import preprocess_style, preprocess_content, denormalize
from postprocessing import postprocess
from nst import stylize
from vgg import load_vgg

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nst")

JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# input constraints (spec §5.2)
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
MIN_DIM, MAX_DIM = 256, 2048

# preset character sets
PRESETS = {
    "uppercase":         "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "uppercase_digits":  "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "letters":           "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
    "alphanumeric":      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789",
}


@dataclass
class JobState:
    status: str = "queued"          # queued | processing | complete | error
    total_chars: int = 0
    completed_chars: int = 0
    current_char: Optional[str] = None
    current_iter: int = 0
    total_iter: int = 0
    error_message: Optional[str] = None
    chars: list[str] = field(default_factory=list)


jobs: dict[str, JobState] = {}

app = FastAPI(title="NST Font Stylizer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"device: {device}")
vgg = load_vgg(device)
log.info("vgg-19 loaded (avg-pool swap, frozen)")


@app.get("/fonts")
def get_fonts():
    return {"fonts": list_fonts()}


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


def _resolve_chars(charset_key: str, custom: str) -> list[str]:
    # turn a preset key OR a custom string into a list of unique chars
    if charset_key == "custom":
        seen = []
        for c in custom:
            if c.strip() and c not in seen:
                seen.append(c)
        return seen
    if charset_key not in PRESETS:
        raise HTTPException(400, f"unknown charset: {charset_key}")
    return list(PRESETS[charset_key])


def _run_job(
    job_id: str, font: str, chars: list[str],
    alpha: float, beta: float, iters: int,
    style_path: Path,
):
    # background task: run pipeline per char, persist results
    state = jobs[job_id]
    try:
        state.status = "processing"
        state.total_chars = len(chars)
        state.total_iter = iters
        log.info(f"[{job_id}] start font={font} chars={len(chars)} iters={iters} α={alpha} β={beta}")

        # preprocess style once, reuse across all glyphs
        style_pil = Image.open(style_path)
        style_t = preprocess_style(style_pil, device)

        out_dir = JOBS_DIR / job_id
        out_dir.mkdir(exist_ok=True)

        for char in chars:
            state.current_char = char
            state.current_iter = 0

            # render glyph + DIP preprocess
            glyph_pil = render_glyph(char, font)
            content_t = preprocess_content(glyph_pil, device)

            # progress callback updates intra-glyph iter counter
            def cb(i: int):
                state.current_iter = i

            # NST + DIP postprocess
            G = stylize(content_t, style_t, vgg, alpha=alpha, beta=beta, iters=iters, progress_cb=cb)
            from torchvision.transforms.functional import to_pil_image
            stylized_pil = to_pil_image(denormalize(G).squeeze(0).cpu())
            final_pil = postprocess(stylized_pil, style_pil)

            # safe filename (handles symbols if user uses custom)
            safe = char if char.isalnum() else f"u{ord(char):04x}"
            final_pil.save(out_dir / f"{safe}.png")

            state.completed_chars += 1
            state.chars.append(char)

        state.current_char = None
        state.current_iter = 0
        state.status = "complete"
        log.info(f"[{job_id}] complete")
    except Exception as e:
        log.exception(f"[{job_id}] error")
        state.status = "error"
        state.error_message = str(e)


@app.post("/stylize")
async def stylize_endpoint(
    background: BackgroundTasks,
    style_image: UploadFile = File(...),
    font: str = Form(...),
    charset: str = Form("uppercase"),
    custom: str = Form(""),
    alpha_beta_ratio: float = Form(1e-4),
    iterations: int = Form(300),
):
    # validate font
    if font not in list_fonts():
        raise HTTPException(400, f"unknown font: {font}")

    # validate file type
    if style_image.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"unsupported type: {style_image.content_type}")

    # read+validate size
    data = await style_image.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 5MB)")

    # validate dims
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if min(w, h) < MIN_DIM:
        raise HTTPException(400, f"image too small (min {MIN_DIM}px short side)")
    if max(w, h) > MAX_DIM:
        # downscale instead of reject - keep aspect ratio
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # validate params
    if not (1e-5 <= alpha_beta_ratio <= 1e-1):
        raise HTTPException(422, "alpha_beta_ratio out of range")
    if not (50 <= iterations <= 1000):
        raise HTTPException(422, "iterations out of range")

    chars = _resolve_chars(charset, custom)
    if not chars:
        raise HTTPException(400, "no characters selected")

    # α=1 fixed, β derived from ratio (spec §7)
    alpha = 1.0
    beta = 1.0 / alpha_beta_ratio

    # register job + persist style image
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    style_path = job_dir / "style.png"
    img.save(style_path)
    jobs[job_id] = JobState()

    # spawn background task
    background.add_task(_run_job, job_id, font, chars, alpha, beta, iterations, style_path)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    s = jobs[job_id]
    # smooth progress: completed glyphs + fractional progress on current glyph
    if s.total_chars > 0:
        frac = (s.current_iter / s.total_iter) if (s.total_iter and s.status == "processing") else 0.0
        progress = (s.completed_chars + frac) / s.total_chars * 100.0
    else:
        progress = 0.0
    return {
        "status": s.status,
        "progress": round(progress, 1),
        "current_char": s.current_char,
        "current_iter": s.current_iter,
        "total_iter": s.total_iter,
        "completed_chars": s.completed_chars,
        "total_chars": s.total_chars,
        "error_message": s.error_message,
    }
```

- [ ] **Step 10.2: Manual verify (`/stylize` happy path on 1 char)**

Start server: `uvicorn main:app --reload`

In another terminal:
```bash
curl -F "style_image=@/tmp/test_style.jpg" \
     -F "font=Roboto" \
     -F "charset=custom" \
     -F "custom=A" \
     -F "alpha_beta_ratio=0.0001" \
     -F "iterations=50" \
     http://localhost:8000/stylize
```
Expected: returns `{"job_id":"<uuid>"}`. Save the uuid.

Poll:
```bash
curl http://localhost:8000/status/<uuid>
```
Expected: progress climbs from 0 → 100 over a minute or two. Server log shows `[<uuid>] start ...` then `complete`. After completion `backend/jobs/<uuid>/A.png` exists.

Stop server.

- [ ] **Step 10.3: Commit**

```bash
git add backend/main.py
git commit -m "feat: /stylize and /status endpoints with background job + dip pipeline"
```

---

## Task 11: Result endpoints (manifest, static png, zip)

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 11.1: Append result endpoints to `backend/main.py`**

Add at the end of `main.py`:

```python
@app.get("/result/{job_id}")
def result_manifest(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    s = jobs[job_id]
    if s.status != "complete":
        raise HTTPException(425, f"not ready (status={s.status})")
    return {
        "chars": s.chars,
        "urls": [f"/result/{job_id}/{(c if c.isalnum() else f'u{ord(c):04x}')}.png" for c in s.chars],
    }


@app.get("/result/{job_id}/{name}")
def result_file(job_id: str, name: str):
    # serve a single stylized png
    p = JOBS_DIR / job_id / name
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/png")


@app.get("/result/{job_id}/zip")
def result_zip(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    s = jobs[job_id]
    if s.status != "complete":
        raise HTTPException(425, "not ready")

    # streaming zip of all stylized pngs
    job_dir = JOBS_DIR / job_id

    def gen():
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
            for c in s.chars:
                safe = c if c.isalnum() else f"u{ord(c):04x}"
                p = job_dir / f"{safe}.png"
                if p.exists():
                    zf.write(p, arcname=f"{safe}.png")
        buf.seek(0)
        yield buf.read()

    headers = {"Content-Disposition": f'attachment; filename="stylized_{job_id[:8]}.zip"'}
    return StreamingResponse(gen(), media_type="application/zip", headers=headers)
```

**Note:** Route order matters in FastAPI. `result_zip` MUST come after the `result_file` route OR be defined before it; here it's after but the path differs (`/result/{job_id}/zip` vs `/result/{job_id}/{name}`). Test in step 11.2 — if zip route gets shadowed, move `result_zip` definition above `result_file`.

- [ ] **Step 11.2: Manual verify**

Start server: `uvicorn main:app --reload`

Re-run a stylize job (Task 10.2) end-to-end, get `<uuid>`, then:
```bash
curl http://localhost:8000/result/<uuid>
curl -o /tmp/A_out.png http://localhost:8000/result/<uuid>/A.png
curl -o /tmp/out.zip http://localhost:8000/result/<uuid>/zip
unzip -l /tmp/out.zip
```
Expected:
- Manifest returns `{"chars":["A"], "urls":["/result/<uuid>/A.png"]}`
- `/tmp/A_out.png` is valid PNG (eyeball it)
- `/tmp/out.zip` lists `A.png`

If `/result/<uuid>/zip` returns 404 (route shadowed by `{name}`), reorder: move `result_zip` definition **above** `result_file` in `main.py` and retest.

- [ ] **Step 11.3: Commit**

```bash
git add backend/main.py
git commit -m "feat: result endpoints - manifest, static png, streaming zip"
```

---

## Task 12: Frontend scaffold (Vite + React + Tailwind)

**Files:**
- Create: `frontend/package.json`, `frontend/index.html`, `frontend/vite.config.js`, `frontend/tailwind.config.js`, `frontend/postcss.config.js`, `frontend/src/main.jsx`, `frontend/src/App.jsx`, `frontend/src/index.css`

- [ ] **Step 12.1: Scaffold with Vite**

```bash
cd frontend
npm create vite@latest . -- --template react
# accept overwrite prompts; pick javascript (not TS)
npm install
npm install -D tailwindcss@3.4.13 postcss@8.4.47 autoprefixer@10.4.20
npx tailwindcss init -p
```

- [ ] **Step 12.2: Configure Tailwind in `frontend/tailwind.config.js`**

```js
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: { extend: {} },
  plugins: [],
};
```

- [ ] **Step 12.3: Replace `frontend/src/index.css`**

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

html, body, #root { height: 100%; }
body { @apply bg-neutral-900 text-neutral-100; }
```

- [ ] **Step 12.4: Replace `frontend/src/App.jsx` with placeholder**

```jsx
export default function App() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <h1 className="text-3xl font-bold">NST Font Stylizer</h1>
    </div>
  );
}
```

- [ ] **Step 12.5: Manual verify**

```bash
cd frontend && npm run dev
```
Open `http://localhost:5173`. Expected: dark page, white "NST Font Stylizer" heading, no console errors. Stop with Ctrl+C.

- [ ] **Step 12.6: Commit**

```bash
git add frontend/
git commit -m "chore: scaffold frontend - vite + react + tailwind"
```

---

## Task 13: API client

**Files:**
- Create: `frontend/src/api.js`

- [ ] **Step 13.1: Write `frontend/src/api.js`**

```js
const BASE = "http://localhost:8000";

export async function getFonts() {
  const r = await fetch(`${BASE}/fonts`);
  if (!r.ok) throw new Error("fonts fetch failed");
  return (await r.json()).fonts;
}

export async function postStylize({ styleFile, font, charset, custom, ratio, iterations }) {
  const fd = new FormData();
  fd.append("style_image", styleFile);
  fd.append("font", font);
  fd.append("charset", charset);
  fd.append("custom", custom);
  fd.append("alpha_beta_ratio", String(ratio));
  fd.append("iterations", String(iterations));
  const r = await fetch(`${BASE}/stylize`, { method: "POST", body: fd });
  if (!r.ok) throw new Error(`stylize failed: ${r.status}`);
  return (await r.json()).job_id;
}

export async function getStatus(jobId) {
  const r = await fetch(`${BASE}/status/${jobId}`);
  if (!r.ok) throw new Error("status failed");
  return await r.json();
}

export async function getManifest(jobId) {
  const r = await fetch(`${BASE}/result/${jobId}`);
  if (!r.ok) throw new Error("manifest failed");
  return await r.json();
}

export function resultUrl(path) {
  return `${BASE}${path}`;
}

export function zipUrl(jobId) {
  return `${BASE}/result/${jobId}/zip`;
}
```

- [ ] **Step 13.2: Commit**

```bash
git add frontend/src/api.js
git commit -m "feat: frontend api client"
```

---

## Task 14: Frontend components — inputs

**Files:**
- Create: `frontend/src/components/StyleUploader.jsx`
- Create: `frontend/src/components/FontSelector.jsx`
- Create: `frontend/src/components/CharSetSelector.jsx`
- Create: `frontend/src/components/ParamSliders.jsx`

- [ ] **Step 14.1: `StyleUploader.jsx`**

```jsx
import { useState } from "react";

export default function StyleUploader({ onChange }) {
  const [preview, setPreview] = useState(null);

  function handleFile(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setPreview(URL.createObjectURL(f));
    onChange(f);
  }

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Style image</label>
      <input
        type="file"
        accept="image/png,image/jpeg,image/webp"
        onChange={handleFile}
        className="block w-full text-sm file:mr-3 file:px-3 file:py-1.5 file:rounded file:border-0 file:bg-neutral-700 file:text-neutral-100 hover:file:bg-neutral-600"
      />
      {preview && (
        <img src={preview} alt="style preview" className="mt-2 max-h-40 rounded border border-neutral-700" />
      )}
    </div>
  );
}
```

- [ ] **Step 14.2: `FontSelector.jsx`**

```jsx
import { useEffect, useState } from "react";
import { getFonts } from "../api";

export default function FontSelector({ value, onChange }) {
  const [fonts, setFonts] = useState([]);

  useEffect(() => {
    getFonts().then(setFonts).catch(console.error);
  }, []);

  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Font</label>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1.5 text-sm"
      >
        <option value="">— select —</option>
        {fonts.map((f) => <option key={f} value={f}>{f}</option>)}
      </select>
    </div>
  );
}
```

- [ ] **Step 14.3: `CharSetSelector.jsx`**

```jsx
const PRESETS = [
  { key: "uppercase",        label: "A-Z" },
  { key: "uppercase_digits", label: "A-Z + 0-9" },
  { key: "letters",          label: "A-Z + a-z" },
  { key: "alphanumeric",     label: "A-Z + a-z + 0-9" },
  { key: "custom",           label: "Custom" },
];

export default function CharSetSelector({ charset, custom, onCharsetChange, onCustomChange }) {
  return (
    <div className="space-y-2">
      <label className="block text-sm font-medium">Characters</label>
      <select
        value={charset}
        onChange={(e) => onCharsetChange(e.target.value)}
        className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1.5 text-sm"
      >
        {PRESETS.map((p) => <option key={p.key} value={p.key}>{p.label}</option>)}
      </select>
      {charset === "custom" && (
        <input
          type="text"
          value={custom}
          onChange={(e) => onCustomChange(e.target.value)}
          placeholder="ABC123"
          className="w-full bg-neutral-800 border border-neutral-700 rounded px-2 py-1.5 text-sm"
        />
      )}
    </div>
  );
}
```

- [ ] **Step 14.4: `ParamSliders.jsx`**

```jsx
export default function ParamSliders({ ratio, iterations, onRatioChange, onIterChange }) {
  // ratio slider works in log-space: slider=0..100 -> ratio=10^(-5 + slider*4/100)
  const sliderVal = ((Math.log10(ratio) + 5) / 4) * 100;

  function setFromSlider(v) {
    const r = Math.pow(10, -5 + (v / 100) * 4);
    onRatioChange(r);
  }

  return (
    <div className="space-y-4">
      <div>
        <div className="flex justify-between text-sm">
          <span>Style ↔ Content (α/β ratio)</span>
          <span className="text-neutral-400">{ratio.toExponential(1)}</span>
        </div>
        <input
          type="range" min={0} max={100} step={1}
          value={sliderVal}
          onChange={(e) => setFromSlider(Number(e.target.value))}
          className="w-full"
        />
        <div className="flex justify-between text-xs text-neutral-500">
          <span>more style</span><span>more content</span>
        </div>
      </div>
      <div>
        <div className="flex justify-between text-sm">
          <span>Iterations</span>
          <span className="text-neutral-400">{iterations}</span>
        </div>
        <input
          type="range" min={100} max={600} step={10}
          value={iterations}
          onChange={(e) => onIterChange(Number(e.target.value))}
          className="w-full"
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 14.5: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: frontend input components"
```

---

## Task 15: Frontend components — submit, progress, results

**Files:**
- Create: `frontend/src/components/StylizeButton.jsx`
- Create: `frontend/src/components/ProgressBar.jsx`
- Create: `frontend/src/components/ResultGrid.jsx`

- [ ] **Step 15.1: `StylizeButton.jsx`**

```jsx
export default function StylizeButton({ disabled, onClick }) {
  return (
    <button
      disabled={disabled}
      onClick={onClick}
      className="w-full px-4 py-2 rounded bg-emerald-600 hover:bg-emerald-500 disabled:bg-neutral-700 disabled:cursor-not-allowed font-medium"
    >
      Stylize
    </button>
  );
}
```

- [ ] **Step 15.2: `ProgressBar.jsx`**

```jsx
export default function ProgressBar({ status }) {
  if (!status) return null;
  const pct = status.progress ?? 0;
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-sm">
        <span>
          {status.status === "processing" && status.current_char
            ? `Stylizing '${status.current_char}' — ${status.current_iter}/${status.total_iter}`
            : status.status}
        </span>
        <span className="text-neutral-400">{pct.toFixed(1)}%</span>
      </div>
      <div className="w-full h-2 bg-neutral-800 rounded overflow-hidden">
        <div className="h-full bg-emerald-500 transition-[width]" style={{ width: `${pct}%` }} />
      </div>
      {status.status === "error" && status.error_message && (
        <div className="text-sm text-rose-400">{status.error_message}</div>
      )}
    </div>
  );
}
```

- [ ] **Step 15.3: `ResultGrid.jsx`**

```jsx
import { resultUrl, zipUrl } from "../api";

export default function ResultGrid({ jobId, manifest }) {
  if (!manifest) return null;
  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">Stylized glyphs ({manifest.chars.length})</h2>
        <a
          href={zipUrl(jobId)}
          className="text-sm px-3 py-1.5 rounded bg-neutral-800 hover:bg-neutral-700"
        >
          Download all (zip)
        </a>
      </div>
      <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
        {manifest.chars.map((c, i) => (
          <div key={i} className="bg-neutral-800 rounded p-1">
            <img src={resultUrl(manifest.urls[i])} alt={c} className="w-full aspect-square object-contain" />
            <div className="text-xs text-center text-neutral-400 mt-1">{c}</div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 15.4: Commit**

```bash
git add frontend/src/components/
git commit -m "feat: frontend submit/progress/result components"
```

---

## Task 16: Wire `App.jsx` — state, polling, layout

**Files:**
- Modify: `frontend/src/App.jsx`

- [ ] **Step 16.1: Replace `frontend/src/App.jsx`**

```jsx
import { useEffect, useState } from "react";
import StyleUploader from "./components/StyleUploader";
import FontSelector from "./components/FontSelector";
import CharSetSelector from "./components/CharSetSelector";
import ParamSliders from "./components/ParamSliders";
import StylizeButton from "./components/StylizeButton";
import ProgressBar from "./components/ProgressBar";
import ResultGrid from "./components/ResultGrid";
import { postStylize, getStatus, getManifest } from "./api";

export default function App() {
  const [styleFile, setStyleFile] = useState(null);
  const [font, setFont] = useState("");
  const [charset, setCharset] = useState("uppercase");
  const [custom, setCustom] = useState("");
  const [ratio, setRatio] = useState(1e-4);
  const [iterations, setIterations] = useState(300);

  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  // poll /status while job is processing
  useEffect(() => {
    if (!jobId || (status && (status.status === "complete" || status.status === "error"))) return;
    const id = setInterval(async () => {
      try {
        const s = await getStatus(jobId);
        setStatus(s);
        if (s.status === "complete") {
          const m = await getManifest(jobId);
          setManifest(m);
          clearInterval(id);
        } else if (s.status === "error") {
          clearInterval(id);
        }
      } catch (e) {
        console.error(e);
      }
    }, 1500);
    return () => clearInterval(id);
  }, [jobId, status?.status]);

  async function handleSubmit() {
    if (!styleFile || !font) return;
    setSubmitting(true);
    setManifest(null);
    setStatus({ status: "queued", progress: 0 });
    try {
      const id = await postStylize({ styleFile, font, charset, custom, ratio, iterations });
      setJobId(id);
    } catch (e) {
      setStatus({ status: "error", error_message: String(e) });
    } finally {
      setSubmitting(false);
    }
  }

  const canSubmit = !!styleFile && !!font && !submitting && (status?.status !== "processing");

  return (
    <div className="min-h-screen p-6 max-w-5xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold">NST Font Stylizer</h1>

      <div className="grid md:grid-cols-2 gap-6 bg-neutral-800/50 border border-neutral-700 rounded p-4">
        <div className="space-y-4">
          <StyleUploader onChange={setStyleFile} />
          <FontSelector value={font} onChange={setFont} />
          <CharSetSelector
            charset={charset} custom={custom}
            onCharsetChange={setCharset} onCustomChange={setCustom}
          />
        </div>
        <div className="space-y-4">
          <ParamSliders
            ratio={ratio} iterations={iterations}
            onRatioChange={setRatio} onIterChange={setIterations}
          />
          <StylizeButton disabled={!canSubmit} onClick={handleSubmit} />
          <ProgressBar status={status} />
        </div>
      </div>

      <ResultGrid jobId={jobId} manifest={manifest} />
    </div>
  );
}
```

- [ ] **Step 16.2: Manual verify (full end-to-end)**

Terminal 1: `cd backend && uvicorn main:app --reload`
Terminal 2: `cd frontend && npm run dev`

Open `http://localhost:5173`. Walk through:
1. Upload a style image (under 5MB, JPG/PNG)
2. Pick a font (e.g. Roboto)
3. Pick "Custom" charset, enter `AB` (small for fast verification)
4. Set iterations slider to 100 (fast smoke)
5. Click Stylize
6. Watch progress bar update with `Stylizing 'A' — N/100`
7. After completion, grid shows 2 stylized glyphs
8. Click "Download all (zip)" — browser downloads zip

If GPU available, full 26-letter test at default 300 iters. Otherwise a 2-3 char smoke test is enough to verify the pipeline.

Eyeball stylized glyphs — they should clearly read as the chosen letters but with style image colors/textures.

- [ ] **Step 16.3: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat: wire app state, polling, and layout"
```

---

## Task 17: README polish + smoke instructions

**Files:**
- Modify: `README.md`

- [ ] **Step 17.1: Replace `README.md`**

```markdown
# NST Font Stylizer

Local web app that stylizes font glyphs using Neural Style Transfer (Gatys et al. 2016) with a classical DIP pre/post-processing pipeline.

## Architecture

- **Frontend** — React + Vite + Tailwind (port 5173)
- **Backend** — FastAPI + PyTorch (port 8000)
- **Model** — VGG-19 pretrained, frozen, max-pool replaced with avg-pool (paper §2)
- **Optimizer** — L-BFGS on pixel values of generated image
- **DIP preprocess** — histogram equalization (style), Gaussian blur (style), Laplacian sharpening (content)
- **DIP postprocess** — median filter, Butterworth low-pass (FFT), histogram matching to style

## Run

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload
```

First run downloads VGG-19 weights (~548MB) to `~/.cache/torch/`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Notes

- NST is slow on CPU (30s-2min per glyph). GPU strongly recommended.
- Default character set: A-Z (26 chars). Bigger sets via dropdown or custom string.
- Output PNGs persist in `backend/jobs/<job_id>/`.
- Bundled fonts: Merriweather, Roboto, JetBrains Mono, Playfair Display, Bebas Neue, Lobster.
```

- [ ] **Step 17.2: Commit**

```bash
git add README.md
git commit -m "docs: project readme with run instructions"
```

---

## Task 18: Final integration smoke test + tag

- [ ] **Step 18.1: Full A-Z run (only if GPU available, otherwise skip to 18.2)**

With backend + frontend running, do a full A-Z run at default settings (ratio=1e-4, iterations=300). Confirm grid renders 26 stylized glyphs. Confirm zip download contains 26 PNGs.

- [ ] **Step 18.2: Sanity smoke (works on CPU too)**

Run a 3-char custom set (e.g. `ABC`) at iterations=100. Should complete in a few minutes on CPU. Confirm:
- Progress bar smoothly updates
- All 3 output glyphs render in grid
- Zip download works
- Backend log shows job start + complete with no exceptions

- [ ] **Step 18.3: Confirm `.gitignore` excludes runtime artifacts**

Run `git status` after a stylize run. Expected: no `backend/jobs/<uuid>/` files appear as untracked. Only `backend/jobs/.keep` is tracked.

- [ ] **Step 18.4: Tag**

```bash
git tag v0.1.0
```

(Do not push to a remote — local demo per spec.)

---

## Spec Coverage Self-Review

| Spec section | Implemented in |
|---|---|
| §3 Architecture (React + FastAPI) | Tasks 9, 12, 16 |
| §4.1 Glyph renderer | Task 4 |
| §4.2 Style + content preprocessing | Task 6 |
| §4.3 VGG-19 with avg-pool | Task 5 |
| §4.4 NST L-BFGS, Gram, content-init | Task 7 |
| §4.5 Postprocessing (median, Butterworth, hist match) | Task 8 |
| §5.1 Endpoints (POST /stylize, GET /status, /result, /fonts, /zip) | Tasks 9, 10, 11 |
| §5.2 Input constraints | Task 10 |
| §5.3 Job lifecycle | Task 10 |
| §5.4 Job storage | Task 10 |
| §5.5 Concurrency (sequential BackgroundTasks) | Task 10 |
| §5.6 Device handling | Task 9 |
| §5.7 Minimal error handling | Tasks 10, 11 |
| §5.8 Code style (1-line block comments, inline labels) | Tasks 4-11 |
| §6 Frontend components + UX flow | Tasks 12-16 |
| §6.5 6 bundled fonts | Task 3 |
| §7 Hyperparameter defaults | Tasks 7, 10 |
| §9 Manual smoke testing | Tasks 4, 6, 7, 8, 9, 10, 11, 16, 18 |
| §13 Open tuning items (k=0.5, cutoff=0.3) | Used in Tasks 6, 8 (default values, tunable) |

All sections covered. No placeholders. Type/identifier consistency checked: `STYLE_LAYERS`, `CONTENT_LAYER`, `LAYER_INDEX`, `gram_matrix`, `stylize`, `extract_features`, `load_vgg`, `preprocess_style`, `preprocess_content`, `denormalize`, `postprocess`, `render_glyph`, `list_fonts`, `JobState`, `_run_job`, `PRESETS` — all referenced consistently across tasks.

---

## Notes for Implementer

- **No formal tests.** Each task ends with a manual verify step and a commit. Do not introduce pytest/Vitest unless the user explicitly asks.
- **Keep code concise.** No defensive try/except, no over-validation, no abstraction layers. Demo-grade clarity.
- **Comment style for NST/DIP modules:** one short comment line before each major block, plus inline labels for important variables (`# content layer`, `# style layers`, `# Gram matrix`).
- **If GPU unavailable**, all tasks still work — just slow. Use small character sets (1-3 chars) and low iteration counts (50-100) for verification steps; bump to defaults only when GPU is available or for the final demo.
- **Tunable params** with empirical defaults: Laplacian `k=0.5` (in `preprocess_content`), Butterworth `cutoff=0.3, order=2` (in `_butterworth_lowpass`). Adjust during the final demo polish if outputs look off.
