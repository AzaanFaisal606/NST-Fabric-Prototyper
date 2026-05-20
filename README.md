# NST Garment Stylizer

Local web app for garment-on-garment style transfer. Upload two clothing photos
(target + source), segment both garments via SAM2, transfer source's pattern /
fabric onto target's garment with Neural Style Transfer (Gatys et al. 2016),
then composite the stylized garment back over the original target photo.

## Architecture

- **Frontend** — React + Vite + Tailwind (port 5173)
- **Backend** — FastAPI + PyTorch (port 8000)
- **NST model** — VGG-19 pretrained, frozen, max-pool replaced with avg-pool (paper §2)
- **Segmentation** — SAM2 (`sam2_hiera_base_plus`)
- **Optimizer** — L-BFGS on pixel values of generated image
- **DIP preprocess** — Lanczos resize, RGB→LAB, hist-eq on L, Gaussian blur, optional bilateral on target garment, mask-edge Gaussian blur
- **DIP postprocess** — median filter, bilateral (mild), LAB histogram matching (masked to source), YIQ luminance swap, Laplacian sharpening
- **Composite** — alpha blend with Gaussian-blurred mask edge

## Run

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download SAM2 checkpoint into `backend/weights/`:

```bash
mkdir -p weights
curl -L -o weights/sam2_hiera_base_plus.pt \
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt
```

Start the server:

```bash
uvicorn main:app --reload
```

First run also downloads VGG-19 weights (~548MB) to `~/.cache/torch/`.

### Frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Usage

1. Upload a Target image (person wearing the garment to re-style).
2. Click on the garment in the Target canvas. Shift+click to exclude regions
   (e.g. skin showing through). The mask updates after each click. Click "Use mask".
3. Repeat for the Source image (garment with the desired pattern / fabric).
4. Adjust α/β ratio and iteration count if desired. Toggle
   "Target garment has a pattern" if your target garment has a strong print
   you want suppressed before stylization.
5. Click Stylize. The progress bar shows current pipeline stage and NST iterations.
6. When complete, download the result PNG.

## Notes

- NST is slow on CPU (~1–3 min per image at short-side=768). GPU is strongly recommended.
- SAM2 inference: ~0.5 s per click on CPU, ~50 ms on GPU.
- Output PNGs persist in `backend/jobs/<job_id>/output.png`.
