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
