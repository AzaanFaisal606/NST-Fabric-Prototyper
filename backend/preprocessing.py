from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# ImageNet normalization (VGG was trained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
TARGET_SHORT_SIDE = 768


def _resize_short_side(img: Image.Image, short_side: int) -> Image.Image:
    # DIP: Lanczos resize (sampling theory)
    w, h = img.size
    scale = short_side / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    return img.resize((new_w, new_h), Image.LANCZOS)


def preprocess_image(
    img: Image.Image,
    device: torch.device,
    suppress_pattern: bool = False,
    mask: Optional[np.ndarray] = None,
) -> tuple[torch.Tensor, Image.Image]:
    # resize short side -> 768 (Lanczos), keep aspect ratio
    img = img.convert("RGB")
    img = _resize_short_side(img, TARGET_SHORT_SIDE)
    arr = np.array(img)

    # DIP: RGB -> LAB color-space conversion
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)

    # DIP: histogram equalization on L channel
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])

    # LAB -> RGB
    arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # DIP: Gaussian blur (sensor-noise suppression)
    arr = cv2.GaussianBlur(arr, ksize=(0, 0), sigmaX=0.5, sigmaY=0.5)

    # optional pattern suppression inside garment region only
    if suppress_pattern and mask is not None:
        # DIP: bilateral filter (large sigma_color: pattern suppression while preserving folds)
        filtered = cv2.bilateralFilter(arr, d=15, sigmaColor=80, sigmaSpace=15)
        m3 = np.clip(mask, 0.0, 1.0).astype(np.float32)[..., None]
        arr = (filtered.astype(np.float32) * m3 + arr.astype(np.float32) * (1.0 - m3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    processed_pil = Image.fromarray(arr)

    # PIL -> normalized [1,3,H,W] tensor on device
    t = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    tensor = t(processed_pil).unsqueeze(0).to(device)
    return tensor, processed_pil


def preprocess_mask(mask: np.ndarray, target_short_side: int = TARGET_SHORT_SIDE) -> np.ndarray:
    # resize binary mask to match image short-side, return float [0,1]
    h, w = mask.shape[:2]
    scale = target_short_side / min(w, h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(mask.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0)


def mask_to_tensor(mask: np.ndarray, device: torch.device) -> torch.Tensor:
    # [H,W] float numpy -> [1,1,H,W] float tensor on device
    m = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return m.to(device)


def soft_mask_for_composite(mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    # DIP: Gaussian blur on mask edge (soft alpha for clean composite)
    m = np.clip(mask.astype(np.float32), 0.0, 1.0)
    blurred = cv2.GaussianBlur(m, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    return np.clip(blurred, 0.0, 1.0)


def denormalize(t: torch.Tensor) -> torch.Tensor:
    # invert ImageNet normalization, clamp to [0,1] for display/save
    mean = torch.tensor(IMAGENET_MEAN, device=t.device).view(1, 3, 1, 1)
    std  = torch.tensor(IMAGENET_STD,  device=t.device).view(1, 3, 1, 1)
    return (t * std + mean).clamp(0, 1)
