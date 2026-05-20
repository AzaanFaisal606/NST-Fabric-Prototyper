import cv2
import numpy as np
from PIL import Image


def composite_back(
    stylized_pil: Image.Image,
    original_target_pil: Image.Image,
    original_mask: np.ndarray,
    mask_blur_sigma: float = 2.0,
) -> Image.Image:
    # resize stylized to original target size (Lanczos)
    target = original_target_pil.convert("RGB")
    stylized = stylized_pil.convert("RGB").resize(target.size, Image.LANCZOS)

    # resize mask to original target size and clamp [0,1]
    w, h = target.size
    mask = cv2.resize(original_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)

    # DIP: soft alpha mask for seamless composite
    mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=mask_blur_sigma, sigmaY=mask_blur_sigma)
    mask3 = np.repeat(mask[..., None], 3, axis=2)

    # alpha blend
    s = np.array(stylized).astype(np.float32)
    t = np.array(target).astype(np.float32)
    out = np.clip(s * mask3 + t * (1.0 - mask3), 0, 255).astype(np.uint8)
    return Image.fromarray(out)
