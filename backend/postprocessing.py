import cv2
import numpy as np
from PIL import Image

# RGB <-> YIQ matrices (NTSC standard)
_RGB2YIQ = np.array([
    [0.299,  0.587,  0.114],
    [0.596, -0.274, -0.322],
    [0.211, -0.523,  0.312],
], dtype=np.float32)
_YIQ2RGB = np.linalg.inv(_RGB2YIQ).astype(np.float32)


def _match_histograms_lab(stylized_arr: np.ndarray, source_arr: np.ndarray, source_mask: np.ndarray) -> np.ndarray:
    # DIP: histogram matching (per-LAB-channel, masked to source garment)
    stylized_lab = cv2.cvtColor(stylized_arr, cv2.COLOR_RGB2LAB)
    source_lab   = cv2.cvtColor(source_arr,   cv2.COLOR_RGB2LAB)
    valid = source_mask > 0.5
    out = stylized_lab.copy()

    # per-channel CDF match using np.interp
    for c in range(3):
        src_pixels = source_lab[..., c][valid]
        if src_pixels.size == 0:
            continue
        sty = stylized_lab[..., c].ravel()
        # source CDF
        s_vals, s_counts = np.unique(src_pixels, return_counts=True)
        s_cdf = np.cumsum(s_counts).astype(np.float64)
        s_cdf /= s_cdf[-1]
        # stylized CDF
        t_vals, t_idx, t_counts = np.unique(sty, return_inverse=True, return_counts=True)
        t_cdf = np.cumsum(t_counts).astype(np.float64)
        t_cdf /= t_cdf[-1]
        # map stylized values -> source values via CDF interpolation
        mapped_vals = np.interp(t_cdf, s_cdf, s_vals)
        out[..., c] = mapped_vals[t_idx].reshape(stylized_lab[..., c].shape).astype(np.uint8)

    return cv2.cvtColor(out, cv2.COLOR_LAB2RGB)


def _yiq_swap(stylized_arr: np.ndarray, content_arr: np.ndarray) -> np.ndarray:
    # DIP: YIQ luminance swap (hard lum-lock; preserves drape from target)
    s = stylized_arr.astype(np.float32) / 255.0
    c = content_arr.astype(np.float32) / 255.0
    s_yiq = s @ _RGB2YIQ.T
    c_yiq = c @ _RGB2YIQ.T
    s_yiq[..., 0] = c_yiq[..., 0]   # replace Y with content luminance
    out = s_yiq @ _YIQ2RGB.T
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ==== IMAGE POSTPROCESSING  (README: Shared components → Image postprocessing) ====
def postprocess(
    stylized_pil: Image.Image,
    content_processed_pil: Image.Image,
    source_processed_pil: Image.Image,
    source_mask: np.ndarray,
) -> Image.Image:
    arr = np.array(stylized_pil.convert("RGB"))

    # clean L-BFGS speckle (salt-and-pepper noise)
    arr = cv2.medianBlur(arr, ksize=3)

    # mild edge-preserving denoise of NST artifacts
    arr = cv2.bilateralFilter(arr, d=5, sigmaColor=25, sigmaSpace=5)

    # remap stylized image's LAB histogram toward the source's actual fabric colours
    source_arr = np.array(source_processed_pil.convert("RGB"))
    arr = _match_histograms_lab(arr, source_arr, source_mask)

    # YIQ Y-swap — copy target's brightness back so folds and drape stay visible
    content_arr = np.array(content_processed_pil.convert("RGB"))
    arr = _yiq_swap(arr, content_arr)

    # Laplacian sharpening — restore a little crispness lost to the blurring above
    lap = cv2.Laplacian(arr, cv2.CV_32F, ksize=3)
    arr = np.clip(arr.astype(np.float32) - 0.3 * lap, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)
