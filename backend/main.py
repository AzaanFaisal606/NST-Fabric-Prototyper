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


@app.get("/result/{job_id}/{name}")
def result_file(job_id: str, name: str):
    # serve a single stylized png
    p = JOBS_DIR / job_id / name
    if not p.exists() or not p.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/png")
