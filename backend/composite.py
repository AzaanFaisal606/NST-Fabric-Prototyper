import cv2
import numpy as np
from PIL import Image


# ==== COMPOSITE  (README: Shared components → Composite) ====
# paste the styled garment region back over the original photo so face / hair / background are untouched
def composite_back(
    stylized_pil: Image.Image,
    original_target_pil: Image.Image,
    original_mask: np.ndarray,
    mask_blur_sigma: float = 2.0,
) -> Image.Image:
    # match stylized image size to the (possibly downsampled) original
    target = original_target_pil.convert("RGB")
    stylized = stylized_pil.convert("RGB").resize(target.size, Image.LANCZOS)

    # bring the garment mask to the same canvas size
    w, h = target.size
    mask = cv2.resize(original_mask.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    mask = np.clip(mask, 0.0, 1.0)

    # soft mask edges = no visible seam between styled garment and original photo
    mask = cv2.GaussianBlur(mask, ksize=(0, 0), sigmaX=mask_blur_sigma, sigmaY=mask_blur_sigma)
    mask3 = np.repeat(mask[..., None], 3, axis=2)

    # inside mask = stylized; outside mask = original target pixels
    s = np.array(stylized).astype(np.float32)
    t = np.array(target).astype(np.float32)
    out = np.clip(s * mask3 + t * (1.0 - mask3), 0, 255).astype(np.uint8)
    return Image.fromarray(out)
