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
