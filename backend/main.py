import io
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from segmentation import load_sam2, segment_with_clicks
from preprocessing import (
    preprocess_image, preprocess_mask, mask_to_tensor, denormalize,
    TARGET_SHORT_SIDE, COARSE_SHORT_SIDE,
)
from postprocessing import postprocess
from composite import composite_back
from nst import stylize
from vgg import load_vgg

# Coarse-to-fine split (Gatys 2017 §6.2). Total iters split between the two stages.
COARSE_FRACTION = 0.4              # 40% coarse @ 384, 60% fine @ 768

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("nst")

# job artifacts persisted under backend/jobs/<job_id>/
JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# input constraints
MAX_UPLOAD_BYTES = 8 * 1024 * 1024
ALLOWED_TYPES = {"image/png", "image/jpeg", "image/webp"}
MIN_DIM, MAX_DIM = 64, 4096


@dataclass
class JobState:
    status: str = "queued"               # queued | processing | complete | error
    stage: Optional[str] = None          # preprocess | nst | postprocess | composite
    current_iter: int = 0
    total_iter: int = 0
    error_message: Optional[str] = None


jobs: dict[str, JobState] = {}

app = FastAPI(title="NST Garment Stylizer")

# allow Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# device + model load (once at startup)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"device: {device}")
vgg = load_vgg(device)                                # MODEL: VGG-19 (frozen, avg-pool swap)
log.info("vgg-19 loaded")
sam2_predictor = load_sam2(device)                    # MODEL: SAM2 (sam2_hiera_base_plus)
log.info("sam2 loaded")


# ---------------------------------------------------------------- helpers


def _load_image(data: bytes, content_type: Optional[str]) -> Image.Image:
    # validate type, decode, validate dims
    if content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"unsupported type: {content_type}")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "file too large (max 8MB)")
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if min(w, h) < MIN_DIM:
        raise HTTPException(400, f"image too small (min {MIN_DIM}px short side)")
    if max(w, h) > MAX_DIM:
        scale = MAX_DIM / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img.convert("RGB")


def _load_mask_png(data: bytes, expected_size: tuple[int, int]) -> np.ndarray:
    # mask file -> bool numpy array at expected (w,h)
    img = Image.open(io.BytesIO(data)).convert("L")
    if img.size != expected_size:
        img = img.resize(expected_size, Image.NEAREST)
    arr = np.array(img)
    return (arr > 127).astype(np.uint8)


# ---------------------------------------------------------------- /health


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": str(device),
        "vgg_loaded": True,
        "sam2_loaded": sam2_predictor is not None,
    }


# ---------------------------------------------------------------- /segment


@app.post("/segment")
async def segment_endpoint(
    image: UploadFile = File(...),
    points: str = Form(...),
):
    # stateless: image + click points -> mask PNG inline
    data = await image.read()
    pil = _load_image(data, image.content_type)
    np_img = np.array(pil)

    try:
        pts = json.loads(points)
    except Exception:
        raise HTTPException(400, "points must be JSON list of {x,y,label}")
    if not isinstance(pts, list) or not pts:
        raise HTTPException(400, "at least one click point required")

    # SAM2 inference (encode + decode)
    mask = segment_with_clicks(sam2_predictor, np_img, pts)

    # encode mask as single-channel PNG
    mask_pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    buf = io.BytesIO()
    mask_pil.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


# ---------------------------------------------------------------- /stylize


def _run_job(
    job_id: str,
    target_path: Path, source_path: Path,
    target_mask_path: Path, source_mask_path: Path,
    alpha: float, beta: float, iters: int,
    suppress_target_pattern: bool,
):
    # coarse-to-fine pipeline (Gatys 2017 §6.2): preprocess x2 -> NST coarse -> upsample -> NST fine -> postprocess -> composite
    state = jobs[job_id]
    out_dir = JOBS_DIR / job_id
    try:
        state.status = "processing"
        state.total_iter = iters

        # iteration split between coarse and fine stages
        coarse_iters = max(1, int(iters * COARSE_FRACTION))
        fine_iters = max(1, iters - coarse_iters)
        log.info(
            f"[{job_id}] start total_iters={iters} (coarse={coarse_iters}@{COARSE_SHORT_SIDE} fine={fine_iters}@{TARGET_SHORT_SIDE}) "
            f"ratio={alpha/beta:.1e} suppress={suppress_target_pattern}"
        )

        # ---------- PREPROCESS (both stages, fresh from original PIL) ----------
        state.stage = "preprocess"
        target_pil_orig = Image.open(target_path).convert("RGB")
        source_pil_orig = Image.open(source_path).convert("RGB")
        target_mask_orig = _load_mask_png(target_mask_path.read_bytes(), target_pil_orig.size)
        source_mask_orig = _load_mask_png(source_mask_path.read_bytes(), source_pil_orig.size)

        # masks at each stage's NST resolution (used both as numpy and as tensors)
        target_mask_coarse = preprocess_mask(target_mask_orig, COARSE_SHORT_SIDE)
        source_mask_coarse = preprocess_mask(source_mask_orig, COARSE_SHORT_SIDE)
        target_mask_fine   = preprocess_mask(target_mask_orig, TARGET_SHORT_SIDE)
        source_mask_fine   = preprocess_mask(source_mask_orig, TARGET_SHORT_SIDE)

        # DIP preprocessing — independently at each scale (NOT cumulative); fine result reused for postprocess
        target_t_coarse, _ = preprocess_image(
            target_pil_orig, device,
            suppress_pattern=suppress_target_pattern,
            mask=target_mask_coarse,
            short_side=COARSE_SHORT_SIDE,
        )
        source_t_coarse, _ = preprocess_image(source_pil_orig, device, short_side=COARSE_SHORT_SIDE)
        target_t_fine, target_processed_pil = preprocess_image(
            target_pil_orig, device,
            suppress_pattern=suppress_target_pattern,
            mask=target_mask_fine,
            short_side=TARGET_SHORT_SIDE,
        )
        source_t_fine, source_processed_pil = preprocess_image(
            source_pil_orig, device, short_side=TARGET_SHORT_SIDE,
        )

        # mask tensors for each stage
        target_mask_coarse_t = mask_to_tensor(target_mask_coarse, device)
        source_mask_coarse_t = mask_to_tensor(source_mask_coarse, device)
        target_mask_fine_t   = mask_to_tensor(target_mask_fine, device)
        source_mask_fine_t   = mask_to_tensor(source_mask_fine, device)

        # ---------- NST COARSE (short-side 384, init=noise) ----------
        state.stage = "nst"

        def cb_coarse(i: int):
            state.current_iter = i

        G_coarse = stylize(
            target_t_coarse, source_t_coarse, vgg,
            alpha=alpha, beta=beta, iters=coarse_iters,
            progress_cb=cb_coarse,
            init_noise=True,
            content_mask=target_mask_coarse_t,
            style_mask=source_mask_coarse_t,
        )

        # ---------- UPSAMPLE BRIDGE (coarse -> fine resolution, bicubic on normalized tensor) ----------
        fine_h, fine_w = target_t_fine.shape[2], target_t_fine.shape[3]
        G_upsampled = F.interpolate(G_coarse, size=(fine_h, fine_w), mode="bicubic", align_corners=False)

        # ---------- NST FINE (short-side 768, init=upsampled G_coarse) ----------
        def cb_fine(i: int):
            state.current_iter = coarse_iters + i        # cumulative across both stages

        G_fine = stylize(
            target_t_fine, source_t_fine, vgg,
            alpha=alpha, beta=beta, iters=fine_iters,
            progress_cb=cb_fine,
            init_image=G_upsampled,
            content_mask=target_mask_fine_t,
            style_mask=source_mask_fine_t,
        )
        stylized_pil = to_pil_image(denormalize(G_fine).squeeze(0).cpu())

        # ---------- POSTPROCESS (fine resolution) ----------
        state.stage = "postprocess"
        final_nst_res = postprocess(
            stylized_pil,
            content_processed_pil=target_processed_pil,
            source_processed_pil=source_processed_pil,
            source_mask=source_mask_fine,
        )

        # ---------- COMPOSITE (at fine NST resolution — no upscale back to native) ----------
        state.stage = "composite"
        target_at_fine = target_pil_orig.resize((fine_w, fine_h), Image.LANCZOS)
        composited = composite_back(
            final_nst_res,
            original_target_pil=target_at_fine,
            original_mask=target_mask_fine.astype(np.float32),
            mask_blur_sigma=2.0,
        )

        composited.save(out_dir / "output.png")
        state.stage = None
        state.status = "complete"
        log.info(f"[{job_id}] complete output_size={composited.size}")
    except Exception as e:
        log.exception(f"[{job_id}] error")
        state.status = "error"
        state.error_message = str(e)


@app.post("/stylize")
async def stylize_endpoint(
    background: BackgroundTasks,
    target_image: UploadFile = File(...),
    source_image: UploadFile = File(...),
    target_mask: UploadFile = File(...),
    source_mask: UploadFile = File(...),
    alpha_beta_ratio: float = Form(1e-4),
    iterations: int = Form(500),
    suppress_target_pattern: str = Form("false"),
):
    # validate images
    target_data = await target_image.read()
    source_data = await source_image.read()
    target_pil = _load_image(target_data, target_image.content_type)
    source_pil = _load_image(source_data, source_image.content_type)

    # masks read raw (PNG, decoded later in job)
    target_mask_data = await target_mask.read()
    source_mask_data = await source_mask.read()
    if not target_mask_data or not source_mask_data:
        raise HTTPException(400, "mask file empty")

    # validate params
    if not (1e-5 <= alpha_beta_ratio <= 1e-1):
        raise HTTPException(422, "alpha_beta_ratio out of range")
    if not (100 <= iterations <= 1000):
        raise HTTPException(422, "iterations out of range (100..1000)")

    # α=1 fixed, β derived from ratio
    alpha = 1.0
    beta = 1.0 / alpha_beta_ratio
    suppress = suppress_target_pattern.lower() in ("true", "1", "yes", "on")

    # register job and persist inputs
    job_id = str(uuid.uuid4())
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(exist_ok=True)
    target_path = job_dir / "target.png"
    source_path = job_dir / "source.png"
    target_mask_path = job_dir / "target_mask.png"
    source_mask_path = job_dir / "source_mask.png"
    target_pil.save(target_path)
    source_pil.save(source_path)
    target_mask_path.write_bytes(target_mask_data)
    source_mask_path.write_bytes(source_mask_data)
    jobs[job_id] = JobState()

    # spawn background pipeline
    background.add_task(
        _run_job, job_id,
        target_path, source_path,
        target_mask_path, source_mask_path,
        alpha, beta, iterations, suppress,
    )
    return {"job_id": job_id}


# ---------------------------------------------------------------- /status, /result


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    s = jobs[job_id]
    # progress = current_iter / total_iter while NST runs; else 0 or 100
    if s.status == "complete":
        progress = 100.0
    elif s.status == "processing" and s.stage == "nst" and s.total_iter > 0:
        progress = round(s.current_iter / s.total_iter * 100.0, 1)
    else:
        progress = 0.0
    return {
        "status": s.status,
        "stage": s.stage,
        "progress": progress,
        "current_iter": s.current_iter,
        "total_iter": s.total_iter,
        "error_message": s.error_message,
    }


@app.get("/result/{job_id}")
def result_manifest(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "job not found")
    s = jobs[job_id]
    if s.status != "complete":
        raise HTTPException(425, f"not ready (status={s.status})")
    return {"url": f"/result/{job_id}/output.png"}


@app.get("/result/{job_id}/output.png")
def result_file(job_id: str):
    p = JOBS_DIR / job_id / "output.png"
    if not p.exists():
        raise HTTPException(404, "file not found")
    return FileResponse(p, media_type="image/png")
