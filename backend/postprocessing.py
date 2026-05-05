import numpy as np
from PIL import Image, ImageFilter
from skimage.exposure import match_histograms


def _butterworth_lowpass(arr: np.ndarray, cutoff: float = 0.3, order: int = 2) -> np.ndarray:
    # frequency-domain low-pass via FFT (Butterworth filter, per channel)
    h, w = arr.shape[:2]
    cy, cx = h / 2.0, w / 2.0
    y, x = np.ogrid[:h, :w]
    # normalized radial distance from center, in [0, ~sqrt(2)/2]
    d = np.sqrt(((y - cy) / h) ** 2 + ((x - cx) / w) ** 2)
    # Butterworth transfer function H(u,v) = 1 / (1 + (D/D0)^(2n))
    H = 1.0 / (1.0 + (d / cutoff) ** (2 * order))

    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(arr.shape[2]):
        F = np.fft.fftshift(np.fft.fft2(arr[..., c].astype(np.float32)))
        out[..., c] = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))
    return np.clip(out, 0, 255).astype(np.uint8)


def postprocess(stylized_pil: Image.Image, style_pil: Image.Image) -> Image.Image:
    # median filter (3x3) - removes salt-and-pepper noise from NST optimization
    img = stylized_pil.filter(ImageFilter.MedianFilter(size=3))

    # Butterworth low-pass (frequency domain) - smooths high-freq artifacts
    arr = np.array(img.convert("RGB"))
    arr = _butterworth_lowpass(arr, cutoff=0.3, order=2)

    # histogram matching to style image - re-aligns color distribution to style palette
    style_arr = np.array(style_pil.convert("RGB").resize(img.size, Image.LANCZOS))
    matched = match_histograms(arr, style_arr, channel_axis=-1).astype(np.uint8)

    return Image.fromarray(matched)
