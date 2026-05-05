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
