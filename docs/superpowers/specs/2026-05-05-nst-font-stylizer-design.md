# NST Font Stylizer — Design Spec

**Date:** 2026-05-05
**Project:** Neural Style Transfer Font Stylization (academic — DIP + Data Science course)
**Status:** Design approved, ready for implementation plan

---

## 1. Overview

Web app that generates custom stylized fonts via Neural Style Transfer. User picks base font + uploads style image; backend renders each glyph, runs NST, and returns stylized character images. Pipeline must justify both deep-learning (NST) and classical DIP techniques for academic rubric.

Reference paper: Gatys et al., *Image Style Transfer Using Convolutional Neural Networks* (CVPR 2016). All NST architectural choices follow paper unless noted.

---

## 2. Tech Stack

| Layer | Choice |
|---|---|
| Frontend | React + Vite + Tailwind CSS |
| Backend | FastAPI (Python) |
| Deep Learning | PyTorch |
| Feature Extractor | VGG-19 pretrained (torchvision), avg-pool swap, frozen |
| Image Processing | PIL, OpenCV, scikit-image, scipy |
| Glyph Rendering | PIL `ImageFont` + `ImageDraw` |
| Fonts | Bundled `.ttf` from Google Fonts |

---

## 3. Architecture

```
┌──────────────────────────────────────┐
│ React + Vite + Tailwind (port 5173)  │
│  - Style upload, font dropdown,      │
│    α/β ratio slider, iter slider,    │
│    char-set dropdown                 │
│  - Polls /status, renders grid       │
└────────────┬─────────────────────────┘
             │ HTTP (REST + multipart)
┌────────────▼─────────────────────────┐
│ FastAPI (port 8000)                  │
│  Endpoints:                          │
│   POST /stylize → job_id             │
│   GET  /status/{id}                  │
│   GET  /result/{id} (manifest)       │
│   GET  /result/{id}/{char}.png       │
│   GET  /result/{id}/zip              │
│   GET  /fonts                        │
│  Modules:                            │
│   glyph_renderer.py                  │
│   preprocessing.py                   │
│   vgg.py (avg-pool VGG-19)           │
│   nst.py (L-BFGS loop)               │
│   postprocessing.py                  │
│  In-mem job dict + disk artifacts    │
│  (backend/jobs/{id}/)                │
└──────────────────────────────────────┘
```

---

## 4. Pipeline

```
User input (font + style image + params)
        ↓
Glyph Renderer       — PIL renders char at 512x512, white bg, black glyph
        ↓
DIP Preprocessing    — applied to style + content (glyph)
        ↓
VGG-19 Features      — frozen, avg-pool, content + style features extracted
        ↓
NST Optimization     — L-BFGS gradient descent on pixels of G
        ↓
DIP Postprocessing   — denoise + frequency smoothing + color match
        ↓
PNG output → disk → frontend grid
```

### 4.1 Glyph Renderer (`glyph_renderer.py`)
- 512x512 canvas, white background, black glyph centered
- Auto-fit font size: start 400pt, shrink until char fits within padding
- API: `render_glyph(char, font_path, size=512) -> PIL.Image`

### 4.2 DIP Preprocessing (`preprocessing.py`)

**Style image:**
- Resize + center crop to 512x512 (NST requires content/style same dims for Gram consistency)
- **Histogram equalization (per-channel RGB)** — boosts global contrast, gives VGG richer texture features
- **Gaussian blur (σ=0.5)** — removes sensor noise so NST doesn't latch onto it as "style"

**Content image (glyph):**
- **Laplacian sharpening** — `output = glyph - k * laplacian(glyph)`. Textbook 2nd-derivative sharpener; hardens anti-aliased glyph edges so VGG `conv4_2` content features encode strong structural signal.

Both then normalized with ImageNet mean/std and converted to tensor.

### 4.3 VGG-19 Feature Extractor (`vgg.py`)
- Load `torchvision.models.vgg19(weights=VGG19_Weights.DEFAULT).features`
- **Replace all `MaxPool2d` with `AvgPool2d`** (paper §2: avg-pool yields more appealing results)
- Frozen (`requires_grad_(False)`), eval mode, moved to device
- Device: auto-detect via `torch.cuda.is_available()`
- **Content layer:** `conv4_2` (paper §3.2 — high-level content, properly merges with style)
- **Style layers:** `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1` (paper §3, multi-scale texture)
- **Style layer weights:** `w_l = 1/5` each (paper-canonical, equal contribution across scales). Hardcoded in `nst.py`, not user-exposed.
- VGG-19 module index map: `conv1_1=0, conv2_1=5, conv3_1=10, conv4_1=19, conv4_2=21, conv5_1=28`
- API: `extract_features(x, layers) -> dict[str, tensor]`
- Loaded once at server startup, kept in memory.

**Deviation from paper:** uses standard pretrained VGG-19, not the paper's mean-activation-rescaled weights. Acceptable; document in writeup.

### 4.4 NST Optimization (`nst.py`)

- **Initialization:** `G = content.clone().requires_grad_(True)` (paper Fig 6 — content-init is deterministic and biases toward letter shape)
- **Optimizer:** L-BFGS on `[G]`, `lr=1.0` (paper §3 explicit: "we use L-BFGS, which we found to work best")
- **Gram matrix:** `gram(F) = F @ F.T / (C*H*W)` (normalized)
- **Loss:**
  ```
  L_total = α * L_content + β * L_style
  L_content = MSE(F_G[conv4_2], F_content[conv4_2])
  L_style   = Σ_l  (1/5) * MSE( gram(F_G[l]),  gram(F_style[l]) )    for l in style_layers
  ```
- **Defaults:** α=1 (fixed), β=1e4 (gives α/β = 1e-4, paper's balanced range), iterations=300
- Calls `progress_cb(iter)` each step → updates job state
- API: `stylize(content_t, style_t, vgg, alpha, beta, iters, progress_cb) -> tensor`

### 4.5 DIP Postprocessing (`postprocessing.py`)

Applied to NST output tensor (denormalized → PIL):
- **Median filter (3x3)** — non-linear, removes salt-and-pepper artifacts from NST optimization (paper §4 explicitly mentions noise as known limitation, justifying denoise step)
- **Butterworth low-pass filter (frequency domain, FFT, order=2, cutoff=0.3)** — frequency-domain smoothing, satisfies DIP rubric for transform-domain technique
- **Histogram matching to style image** — `skimage.exposure.match_histograms`. Re-aligns output color distribution to style palette; classical DIP technique.

API: `postprocess(stylized_tensor, style_img) -> PIL.Image`

### 4.6 DIP Techniques Summary (academic justification)

| Stage | Technique | Category |
|---|---|---|
| Style preprocess | Resize + center crop | Geometric transform |
| Style preprocess | Histogram equalization (RGB) | Histogram operation |
| Style preprocess | Gaussian blur | Spatial filter (linear, smoothing) |
| Content preprocess | Laplacian sharpening | Spatial filter (2nd derivative) |
| Core | Gram matrix, MSE feature loss | Statistical / loss |
| Postprocess | Median filter (3x3) | Spatial filter (non-linear) |
| Postprocess | Butterworth low-pass | Frequency domain (FFT) |
| Postprocess | Histogram matching | Histogram operation |

Covers: spatial filtering (linear + non-linear), frequency domain, histogram operations, geometric transforms.

---

## 5. Backend (FastAPI)

### 5.1 Endpoints

**`POST /stylize`** (multipart)
- Body: `style_image` (file), `font` (str), `characters` (str, comma-separated or preset key), `alpha_beta_ratio` (float), `iterations` (int)
- Validates inputs (Pydantic). On success: writes style image to disk, registers job, spawns `BackgroundTasks`, returns `{job_id}` immediately.

**`GET /status/{job_id}`**
- Returns: `{status, progress, current_char, current_iter, total_iter}`
- `status`: `"queued" | "processing" | "complete" | "error"`
- `progress`: `(completed_chars + current_iter/total_iter) / total_chars * 100` (smooth bar)

**`GET /result/{job_id}`**
- Returns JSON manifest: `{chars: ["A", "B", ...], urls: ["/result/{id}/A.png", ...]}`

**`GET /result/{job_id}/{char}.png`**
- Static file serve from `backend/jobs/{id}/{char}.png`

**`GET /result/{job_id}/zip`**
- Streaming zip of all stylized PNGs

**`GET /fonts`**
- Returns: `["Merriweather", "Roboto", "JetBrainsMono", "PlayfairDisplay", "BebasNeue", "Lobster"]`

### 5.2 Input Constraints

- Style image: max 5MB, formats PNG/JPG/WEBP, min 256x256, max 2048x2048 (resize down server-side before pipeline)
- Iterations: 100–600 (slider range)
- α/β ratio: 1e-5 to 1e-1 (log-scale slider, default 1e-4)
- Characters: A-Z (default), or preset {A-Z, A-Z+0-9, A-Z+a-z, A-Z+a-z+0-9}, or custom string

### 5.3 Job Lifecycle

```
POST /stylize
  validate → save style → register job → spawn task → return job_id

[background task]
  status = "processing"
  preprocess style (once, cached)
  for char in chars:
    glyph = render_glyph(char, font)
    glyph_t = preprocess_content(glyph)
    G = nst.stylize(glyph_t, style_t, vgg, α, β, iters, progress_cb)
    out = postprocess(G, style_img)
    out.save(f"jobs/{id}/{char}.png")
    completed_chars += 1
  status = "complete"
  manifest written

GET /status/{id}      → progress payload
GET /result/{id}      → manifest JSON
GET /result/{id}/X.png → static file
GET /result/{id}/zip  → streaming zip
```

### 5.4 Job Storage

- `jobs: dict[str, JobState]` in-memory (lost on restart)
- Result PNGs persisted to `backend/jobs/{id}/` (survive restart but won't be re-indexed without DB; acceptable for demo)
- Style image saved as `backend/jobs/{id}/style.png`
- No cleanup logic — manual delete acceptable for demo

### 5.5 Concurrency

- Single FastAPI worker, sequential `BackgroundTasks`. Multiple jobs serialize naturally; no queue logic. Acceptable for academic demo.

### 5.6 Device Handling

- Auto-detect: `device = "cuda" if torch.cuda.is_available() else "cpu"`
- Portable: deploy on GPU host (CUDA wheel) auto-uses GPU; CPU host falls back. Same code.
- VGG and tensors moved to `device` once at startup.

### 5.7 Error Handling (minimal — demo scope)

- Pydantic validation on `/stylize` params (auto)
- 404 on unknown `job_id`
- Background task wrapped in try/except → log exception + set `jobs[id].status = "error"` (so frontend doesn't hang)
- Python `logging` at INFO level: job start, end, exceptions
- No retry logic, no granular HTTP error mapping, no client-side validation. YAGNI for demo.

### 5.8 Code Style (per user)

- Concise, no overengineering
- 1-line comment before each major block of code in NST and DIP modules, explaining what the block does
- Short inline labels for important lines (e.g. `# content layer`, `# style layers`, `# Gram matrix`)
- Well-formatted, easily readable, easy to explain in a course defense
- Avoid over-decoration; technical depth must remain visible in code

---

## 6. Frontend (React + Vite + Tailwind)

### 6.1 Structure

```
frontend/src/
├── App.jsx                    # layout, holds top-level state
├── main.jsx                   # Vite entry
├── api.js                     # postStylize, getStatus, getResult
├── components/
│   ├── StyleUploader.jsx      # drag-drop / file input + preview
│   ├── FontSelector.jsx       # dropdown, 6 fonts
│   ├── CharSetSelector.jsx    # preset dropdown + custom text input
│   ├── ParamSliders.jsx       # α/β ratio (log) + iterations
│   ├── StylizeButton.jsx      # POST /stylize, kicks off polling
│   ├── ProgressBar.jsx        # bar + "Stylizing 'M' — 250/500"
│   └── ResultGrid.jsx         # CSS grid + Download All
└── index.css                  # Tailwind directives
```

### 6.2 State (App.jsx, no Redux)

- Inputs: `styleFile`, `font`, `charSet`, `customChars`, `ratio`, `iterations`
- Job: `jobId`, `status`, `progress`, `currentChar`, `currentIter`
- Result: `resultManifest` (list of char URLs)

### 6.3 Polling

- `useEffect` with `setInterval(1500ms)` while `status === "processing"`. Clears on `complete` or `error`.

### 6.4 UX Flow

1. Upload style → preview thumbnail
2. Pick font, char-set preset (or custom), set α/β + iter sliders
3. Click "Stylize" → button disables, progress bar appears
4. Bar updates every 1.5s with `progress%` and `"Stylizing 'M' — 250/500 iters"` label
5. On `complete` → fetch manifest → render grid of stylized glyphs
6. "Download All" → `GET /result/{id}/zip` → browser download

### 6.5 Fonts (bundled)

| Family | Style |
|---|---|
| Merriweather | serif |
| Roboto | sans-serif |
| JetBrains Mono | monospace |
| Playfair Display | display serif |
| Bebas Neue | display sans |
| Lobster | script |

All from Google Fonts (free, redistributable). Bundled as `.ttf` in `backend/fonts/`.

### 6.6 Code Style (per user)

- Concise, minimal comments (component names self-document)
- No unused state, no premature abstractions
- Tailwind utility classes, dark/neutral theme, single-page

---

## 7. NST Hyperparameter Reference

| Param | Default | Range | Notes |
|---|---|---|---|
| optimizer | L-BFGS | — | paper-canonical |
| lr | 1.0 | — | L-BFGS standard |
| iterations | 300 | 100–600 | slider |
| α (content weight) | 1 | fixed | paper convention |
| β (style weight) | 1e4 | derived | β = 1 / ratio |
| α/β ratio | 1e-4 | 1e-5 to 1e-1, log | single slider, semantic "more content ↔ more style" |
| content layer | conv4_2 | fixed | paper §3.2 |
| style layers | conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 | fixed | paper §3 |
| style layer weights w_l | 1/5 each | fixed | paper §3 |
| init | content image clone | fixed | paper Fig 6, deterministic |
| pooling | avg-pool (replaces max-pool) | fixed | paper §2 |

---

## 8. Folder Structure

```
project/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.jsx
│   │   ├── api.js
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── index.html
│
├── backend/
│   ├── main.py               # FastAPI app, route definitions, job mgmt
│   ├── nst.py                # NST optimization loop (L-BFGS)
│   ├── vgg.py                # VGG-19 loader (avg-pool) + feature extractor
│   ├── glyph_renderer.py     # PIL-based glyph rendering
│   ├── preprocessing.py      # DIP preprocessing (style + content)
│   ├── postprocessing.py     # DIP postprocessing (median, Butterworth, hist match)
│   ├── fonts/                # bundled .ttf files (6 fonts)
│   ├── jobs/                 # runtime: per-job style image + result PNGs
│   └── requirements.txt
│
├── docs/
│   └── superpowers/specs/
│       └── 2026-05-05-nst-font-stylizer-design.md   # this file
│
├── description.md
└── Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf
```

---

## 9. Testing

Manual smoke tests only. No unit tests, no pytest, no Vitest. YAGNI for academic demo.

**Backend (manual during dev):**
- `GET /fonts` returns 6 fonts
- `glyph_renderer.render_glyph("A", "Roboto")` → eyeball PNG
- `preprocess_style(test.jpg)` returns tensor `[1,3,512,512]`
- End-to-end NST on 1 char with 50 iters → eyeball output PNG
- `POST /stylize` via curl → poll `/status` → fetch `/result`

**Frontend (manual):**
- Upload, submit, watch progress, render grid, download zip — all in browser dev mode.

---

## 10. Scope & Constraints

- Local demo only, no cloud deployment
- NST per glyph: 30s–2min CPU; significantly faster on GPU
- Default char set A-Z (26 chars) for fast demo. User can opt into more.
- VGG-19 weights download on first run (~548MB), cached at `~/.cache/torch/`
- Concurrency: single job serialization is acceptable
- No auth, no rate limiting, no DB

---

## 11. Out of Scope (explicit)

- User accounts / auth
- Persistent job DB
- Cloud deploy / containerization
- Per-glyph parallelism (would need multi-worker FastAPI + model replication)
- Advanced NST variants (AdaIN, fast-NST, etc.) — paper baseline only
- Vector-format output (.ttf / SVG) — raster PNG only
- Auto-cleanup of old jobs
- Formal test suite

---

## 12. Paper-Derived Decisions Recap

| Decision | Paper source |
|---|---|
| L-BFGS optimizer | §3 explicit |
| Content layer `conv4_2` | §3.2, Fig 5 |
| 5 style layers `conv*_1`, w_l = 1/5 | §3 |
| Avg-pool replaces max-pool | §2 |
| α/β ratio framing (default 1e-4) | §3.1, Fig 4 |
| Content-image init (deterministic) | §3.3, Fig 6 |
| 512x512 resolution | §4 |
| Style image resized to content size | §3 |
| Postprocess denoising justified | §4 (acknowledged as known limitation) |

---

## 13. Open Items for Implementation Phase

- Decide on exact Tailwind color palette (defer to implementation, low-stakes)
- Pick exact font sizes / aspect for glyph render auto-fit (tune empirically per font)
- Butterworth cutoff `0.3` is starting value — may need tuning during integration testing
- Laplacian sharpening kernel strength `k` — start with k=0.5, tune empirically
