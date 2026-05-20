

## Insights from Research Paper:
Direct paper specifications:
- Optimizer: L-BFGS — paper §3 explicit: "Here we use L-BFGS, which we found to work best for image synthesis"
- Content layer: conv4_2 — confirmed in §3
- Style layers: conv1_1, conv2_1, conv3_1, conv4_1, conv5_1 with weights w_l = 1/5 each (1/N where N=5 layers used), zero for others — confirmed §3
- α/β ratios used in paper: 1e-3, 1e-4, 5e-4, 8e-4 — α/β ranges from 1e-4 (heavy style) to 1e-1 (heavy content). Default in main results: α/β = 1e-3. Translated to our convention (α=1, β=?): β between 1e3 and 1e4 for balanced result, 1e4 for default.
- Resolution: 512x512 — confirmed §4 ("about 512x512 pixels"). Matches our spec.
- Average pooling preferred over max pooling — paper §2: "replacing the maximum pooling operation by average pooling yields slightly more appealing results". Easy to swap in PyTorch VGG.
- VGG normalization — paper used VGG with weights rescaled so mean activation per filter = 1. Standard torchvision VGG-19 doesn't have this. Mention as deviation from paper — use standard pretrained, document tradeoff.
- Initialization — paper §3.3: white noise standard, but content-image init or style-image init also work and give deterministic output. Content-image init recommended for our case (per description) — gives consistent, deterministic glyph output. Paper confirms valid choice.

New insights worth incorporating:

1. Average pooling swap — replace VGG max-pool with avg-pool layers. One-line fix in PyTorch. Improves quality per paper. Worth doing.
2. Layer-wise style weights w_l — paper sets equal weights 1/5 for all 5 style layers (zero elsewhere). Our spec didn't mention this. Should be in NST loss code. Could expose as advanced param but default 1/5 each.
3. α/β ratio reframe — paper describes ratio, not absolute values. Frontend should ideally show ratio α/β as single slider (semantic: "more content ↔ more style") with absolute α=1 fixed internally. Or keep two sliders but with sensible log defaults.
4. Fig 5 finding — content layer choice matters: conv2_2 keeps too much pixel detail (texture just blended over photo); conv4_2 properly merges content and style. Confirms our choice. Could mention as alternate user knob ("preserve more detail" toggle = use conv2_2) but YAGNI for demo.
5. Loss gradient explicit form — paper eq. 2 and 6 give analytical gradients but PyTorch autograd handles this. No code impact, just academic justification.
6. Resolution warning — paper §4 admits 512x512 takes "up to an hour on Nvidia K40 GPU". Modern GPUs faster, but CPU will be brutal. Already noted in spec.
7. Style image resize — paper §3: "we always resized the style image to the same size as the content image before computing its feature representation". Reinforces our preprocess step.
8. Limitation noted — paper §4: synthesized images sometimes have "low-level noise... it could thus be tempting to construct efficient de-noising techniques to post-process the images". This explicitly justifies our postprocessing stage for the academic writeup.


1. Mask-Guided Content Loss

What: Restrict content loss to certain pixels only, instead of whole image.

Why: Without mask, content loss says "every pixel of G must look like content image at conv4_2." With mask, only garment region is anchored to content; background free.

Math: Standard content loss:

L_content = mean( (F_G - F_content)^2 )

Mask-guided:

L_content = sum( m * (F_G - F_content)^2 ) / sum(m)

m = mask resized to feature-map spatial dims, values 0..1. Pixels with m=0 contribute nothing to gradient.

Implementation (already in your nst.py:60-74):

m = F.interpolate(content_mask, size=(fh, fw), mode="bilinear")
diff = (G_feat - content_target) ** 2
content_loss = (diff * m).mean()   # weighted average

For garment project: content_mask = SAM2 segmentation of target garment. Letter problem becomes shirt problem.

---
2. Mask-Guided Style Loss

What: Sample style Gram matrix only from style image's garment region, ns face, background, skin).

Why: If style photo is "person wearing patterned shirt," default Gatys mtones, wall texture, hair color into one Gram matrix → output bleeds skincolor into target.

Math: Standard Gram:

G_l = F_l F_l^T / (C·H·W)         # F_l shape [C, H*W]

Masked Gram:

F_l_masked = F_l ⊙ m_l            # zero out non-garment columns
G_l = F_l_masked F_l_masked^T / sum(m_l)

Effectively only garment-region feature columns contribute to outer-prodignored.

Implementation sketch:

def masked_gram(feat, mask_feat):
    b, c, h, w = feat.shape
    m = mask_feat.view(b, 1, h*w)              # [1,1,HW]
    flat = feat.view(b, c, h*w) * m            # zero out bg cols
    norm = m.sum().clamp(min=1.0) * c
    return flat @ flat.transpose(1, 2) / norm

Drop-in replacement for gram_matrix(feats[l]) in style loss, both for style target and generated G — done with G's mask too so generator only "tries" to match in foreground
region.

Multi-region extension (Champandard's neural doodle): different masks pellar, body) → separate Gram per region → richer control.

---
3. Shading-Preservation Loss (luminance lock)

What: Force G's brightness map to equal content image's brightness map. Style transfer happens only in chroma (color) channels.

Why: Folds, wrinkles, shadows on target garment = pure luminance variation. If preserved, output looks like same garment in new fabric. Without it, shading washes out and
output looks flat / decal-like.

Math: Convert RGB → YCbCr (or LAB).

Y = 0.299 R + 0.587 G + 0.114 B    (luminance)

Add loss:

L_lum = mean( (Y_G - Y_content)^2 )

Or hard constraint after each iter: replace G's Y channel with content's

Implementation sketch:

def luminance(t):
    # t in [0,1], shape [1,3,H,W]
    r, g, b = t[:, 0:1], t[:, 1:2], t[:, 2:3]
    return 0.299*r + 0.587*g + 0.114*b

# in closure (G is normalized — denormalize first or compute on raw):
G_denorm = denormalize(G)
content_denorm = denormalize(content_t)
lum_loss = F.mse_loss(luminance(G_denorm), luminance(content_denorm))
loss = alpha*content_loss + beta*style_loss + gamma*lum_loss

Alternative (Gatys 2016 "Preserving Color in NST"): YIQ luminance swap. After NST done, convert output to YIQ, replace Y with content's Y, convert back. Simpler, no extra loss
 term.

For garment: shadow under arm, highlight on shoulder, fabric folds = allreserved → realistic prototype.

---
4. SAM2 Masks (Segment Anything 2, Meta 2024)

What: Foundation model for image segmentation. Input image (+ optional prompt: click point, bbox, text). Output: high-quality binary mask per object.

Architecture (brief):
- ViT image encoder (heavy, run once per image).
- Prompt encoder (lightweight): point/bbox/mask → embeddings.
- Mask decoder (lightweight transformer): combines image + prompt embedd
- SAM2 adds memory bank for video (track object across frames). For static images = drop-in upgrade over SAM1, sharper boundaries.

Why use it for garment project:
- User uploads two clothing photos. Need to know "where is shirt in imag
- Old way: train segmentation net on labeled garment dataset. Slow, narrow, fails on unseen styles.
- SAM2 way: foreground-click in UI ("user clicks on shirt") → mask in ~1eneralizes to any garment.

Implementation:

# pip install segment-anything-2  (or git clone Meta repo)
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt")
predictor = SAM2ImagePredictor(sam)

predictor.set_image(np_image)              # ViT encode once
masks, scores, _ = predictor.predict(
    point_coords=np.array([[x, y]]),       # user click on garment
    point_labels=np.array([1]),            # 1 = foreground
    multimask_output=True,
)
best = masks[scores.argmax()]              # H, W bool array

Variants:
- Auto mask generator: no prompt, returns all objects. Good for "find latic.
- Box prompt: drag rectangle over garment.
- Text prompt (via Grounded-SAM, GroundingDINO+SAM): "shirt" → mask. Rem

Output format: boolean numpy array [H, W]. Convert to [1,1,H,W] float ter existing content_mask slot in stylize(). No code change needed in NSTloop, only new preprocess step.

Cost:
- Model weights: ~225MB (large), ~80MB (base+).
- Inference: ~0.5s on CPU per image, ~50ms on GPU.
- Acceptable for two-image prototype tool.

---
How the three combine for garment project

target_image, source_image  → SAM2 → target_mask, source_mask

stylize(
    content_t = target_image,          # garment-on-person photo
    style_t   = source_image,          # shirt with desired pattern
    content_mask = target_mask,        # mask-guided content loss (you h
    style_mask   = source_mask,        # mask-guided style loss (new)
    luminance_lock = True,             # shading-preserve loss (new)
)
→ composite output back over original target_image outside target_mask

That's the whole pipeline. Three losses + one segmentation step. Falls ctylize() closure structure.

Option 1: Strip pattern from content before NST (preprocessing)

Flatten content garment to "albedo + shading only," removing high-frequency pattern:

- Bilateral filter (large σ_color, small σ_spatial) — removes texture, keeps edges/folds. OpenCV cv2.bilateralFilter. Classic DIP technique → fits your DIP course angle.
- Guided filter (He et al. 2010) — similar, sharper.
- Median filter (large kernel) — kills small-scale pattern.

Apply only inside garment mask. Effect: garment becomes solid-color shape with only shading variation. Then content loss anchors to shape + drape, not pattern. Style transfer fills in new pattern cleanly.

Option 3: Luminance lock from de-patterned content

Run lum-lock against bilateral-filtered content, not raw. Folds = low-frequency luminance, kept. Pattern = high-frequency luminance, removed. Best of both.

content_smooth = cv2.bilateralFilter(content_rgb, d=15, sigmaColor=80, sigmaSpace=80)
Y_target = luminance(content_smooth)   # shading only, no pattern
lum_loss = mse(luminance(G), Y_target)

---

# Project Pivot — Garment Style Transfer

The project has been re-scoped from font glyphs to garment-on-garment style transfer.
Users upload two clothing photos: a target image (person wearing the garment to
be re-styled) and a source image (garment with the desired pattern / fabric).
The system segments both garments with SAM2, transfers the source's pattern /
fabric onto the target's garment via NST, and composites the stylized garment
back over the original target photo. Background, face, pose are preserved.

## Pipeline

```
target.jpg, source.jpg
        ↓
[SAM2 segmentation]   — user clicks per image, positive/negative refinement
        ↓                                         ↓
target_mask                                  source_mask
        ↓                                         ↓
[DIP Preprocessing on both images]
        ↓
[VGG-19 Feature Extraction (frozen, avg-pool)]
        ↓
[NST Optimization Loop, mask-guided content + style losses]
        ↓
[DIP Postprocessing on stylized output]
        ↓
[Alpha-blend composite back over target image]
        ↓
output.png
```

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React + Vite + Tailwind |
| Backend | FastAPI (Python) |
| Deep Learning | PyTorch |
| Feature Extractor | VGG-19 (pretrained, frozen, max-pool→avg-pool) |
| Segmentation | SAM2 (sam2_hiera_base_plus) |
| Image Processing | PIL, OpenCV, scikit-image, NumPy |

## Model & Layers (NST)

- **Backbone:** VGG-19 pretrained on ImageNet (`torchvision.models.vgg19`).
  Frozen — no training. Max-pool layers replaced with avg-pool (paper §2).
- **Content layer:** `conv4_2` (paper §3, Fig 5).
- **Style layers:** `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1`,
  equal weights w_l = 1/5 each.
- **Loss:** `α * L_content_masked + β * L_style_masked`, where:
  - Content loss is mask-weighted by the target garment mask (only target-garment
    pixels anchor to content features).
  - Style targets are computed using a masked Gram matrix sampled only inside
    the source garment region (face / skin / background of source ignored).
  - G's style features in the closure are masked by the target mask so that
    only G's target-garment region is constrained to match the source's Gram.
- **Optimizer:** L-BFGS over pixels of the generated image G.
- **Initialization:** Gaussian noise (`randn × 0.01`).
- **Default α/β = 1e-4**, default iterations = 500 (range 100–1000).
- **Resolution:** target image short-side resized to 768 for NST; composite at
  the native target resolution.

## Segmentation (SAM2)

- Model: `sam2_hiera_base_plus` (~309 MB) checkpoint, server-side inference.
- User interaction: click on garment in browser canvas. Positive clicks (normal)
  expand the mask, negative clicks (shift-click) shrink it. Mask refreshes after
  each click.
- Backend endpoint `POST /segment` is stateless — accepts the image plus a JSON
  list of click points and returns the mask PNG inline. The frontend holds the
  mask blob until stylize submit.

## DIP Preprocessing

| Technique | Stage | Purpose |
|---|---|---|
| Resize (Lanczos) | both images | Sampling, normalize to short-side 768 |
| RGB → LAB | both images | Decouple luminance from chroma |
| Histogram equalization (L channel) | both images | Even out lighting |
| Gaussian blur (σ ≈ 0.5) | both images | Sensor-noise suppression |
| Bilateral filter (large σ_color ≈ 80, σ_space ≈ 15) | target only, toggleable | Pattern suppression while preserving folds (Tomasi & Manduchi 1998) |
| Gaussian blur on mask edge (σ ≈ 1) | both masks | Soft alpha for clean composite |

## DIP Postprocessing

| Technique | Purpose |
|---|---|
| Median filter (3×3) | Salt-and-pepper noise from L-BFGS |
| Bilateral filter (mild, σ_color ≈ 25, σ_space ≈ 5) | Edge-preserving denoise of NST artifacts |
| Histogram matching to source garment, per-LAB-channel, masked | Color-distribution alignment to source palette |
| YIQ luminance swap | Hard luminance lock — preserves drape and folds from target |
| Laplacian sharpening (k = 0.3) | Edge restoration after bilateral smoothing |

The bilateral filter is applied at two parameter regimes (large σ in preprocessing
to suppress pattern, small σ in postprocessing to denoise) — same algorithm,
opposite intent, demonstrating parametric control of an edge-aware filter.

## Composite

`final = stylized_upscaled · mask_blur + original_target · (1 − mask_blur)`
where `mask_blur` is the Gaussian-edge-blurred target mask. Background, face,
pose of the original target are preserved exactly.

## Backend Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST /segment` | image + click points → mask PNG (stateless) |
| `POST /stylize` | both images + both masks + params → job_id |
| `GET /status/{job_id}` | progress (stage label + iter counter) |
| `GET /result/{job_id}` | manifest (single output URL) |
| `GET /result/{job_id}/output.png` | final composited image |
| `GET /health` | device, model-loaded flags |

## DIP Techniques Summary (academic justification)

| Stage | Technique |
|---|---|
| Pre | Lanczos resize, RGB→LAB, hist-eq on L, Gaussian blur, bilateral (toggle), mask-edge blur |
| Core | VGG-19 feature extraction, masked Gram-matrix style representation, L-BFGS pixel optimization |
| Post | Median, bilateral (mild), LAB histogram matching, YIQ Y-swap, Laplacian sharpen |
| Composite | Alpha blend with soft mask |

## Notes

- NST per image takes ~1–3 min on CPU at 768², much faster on GPU.
- SAM2 inference ~0.5 s on CPU, ~50 ms on GPU.
- Output persists at `backend/jobs/<job_id>/output.png`.
