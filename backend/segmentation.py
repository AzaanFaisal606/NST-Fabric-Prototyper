from pathlib import Path
import numpy as np
import torch

# ==== SEGMENTATION MODEL  (README: Shared components → Segmentation model) ====
# SAM2 (Segment Anything 2 from Meta) — click-driven garment masks, no custom training
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# config and checkpoint paths
SAM2_CONFIG = "configs/sam2/sam2_hiera_b+.yaml"
DEFAULT_CKPT = Path(__file__).parent / "weights" / "sam2_hiera_base_plus.pt"


def load_sam2(device: torch.device, ckpt_path: Path = DEFAULT_CKPT) -> SAM2ImagePredictor:
    # build SAM2 from config + checkpoint, wrap in image predictor
    if not Path(ckpt_path).exists():
        raise FileNotFoundError(
            f"SAM2 checkpoint not found at {ckpt_path}. "
            "Download sam2_hiera_base_plus.pt from "
            "https://github.com/facebookresearch/sam2 and place it there."
        )
    model = build_sam2(SAM2_CONFIG, str(ckpt_path), device=device)
    model.eval()
    return SAM2ImagePredictor(model)


def segment_with_clicks(
    predictor: SAM2ImagePredictor,
    np_img: np.ndarray,
    points: list[dict],
) -> np.ndarray:
    # click-prompted segmentation; returns highest-score boolean mask
    if len(points) == 0:
        raise ValueError("at least one click point required")

    # heavy step: ViT encodes the image once — every click after this is fast
    predictor.set_image(np_img)

    point_coords = np.array([[p["x"], p["y"]] for p in points], dtype=np.float32)
    point_labels = np.array([p["label"] for p in points], dtype=np.int32)

    # decode mask from click prompts (lightweight)
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    best = int(scores.argmax())
    return masks[best].astype(bool)
