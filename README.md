# NST Fabric Prototyper

## What this project does

A small web app for **digital fabric prototyping**. The user uploads two
clothing photos: a **target** photo (a person wearing the garment they want
to re-style) and a **source** photo (a shirt with the pattern or fabric they
want to copy over). The system clicks-to-select each garment, runs Neural
Style Transfer to repaint the target shirt with the source's pattern, and
finally pastes the new garment back onto the original photo so the person's
face, pose, and background are unchanged.

The codebase is organised so that each section of this README has a matching
heading in the source code. Search the project for `README:` (e.g.
`grep -rn "README:" backend frontend`) and you will land in the file that
implements whichever section you are reading.

---

## Shared components (used by both branches)

These are the building blocks the project always uses, regardless of branch.

### Deep learning model — VGG-19

VGG-19 is a convolutional neural network originally built to recognise
objects in photos. We use it because it has already learned, on millions
of images, what edges, textures, and shapes look like at many different
levels of detail. We do not train it — we just borrow its eyes.

Two practical adjustments:
- The network is **frozen**: its weights never change. Only the image
  we are creating is updated during optimisation.
- The original VGG uses "max-pooling" layers (which keep only the
  loudest signal in every small patch). We replace those with
  "average-pooling" (which keeps the average instead). The original NST
  paper recommends this swap because the resulting textures look
  smoother and more natural for stylisation.

### Content layer

When NST asks "does this picture still show the same thing?", it does
not compare pixel by pixel. Instead it picks one mid-deep layer of
VGG-19, called **conv4_2**, and compares the activations there. At
that depth the network has already abstracted away small textures and
sees the *arrangement* of the picture — where the shirt is, the
silhouette of the person, the folds — but it has not yet abstracted
away so much that it forgets the structure. Using `conv4_2` is the
sweet spot recommended by the original paper.

### Style layers

For "does this picture have the same look and feel?", NST uses **five**
layers at different depths: `conv1_1, conv2_1, conv3_1, conv4_1, conv5_1`.
Think of them as five viewpoints from very fine (the tightest weave of
threads) to very coarse (the overall pattern shape). At each viewpoint
NST computes a small statistic called a **Gram matrix** — basically a
fingerprint of "which features tend to fire together". When we ask the
generated image to match those five fingerprints, the result picks up
the source's style at all those scales at once. All five layers are
weighted equally (1/5 each).

### Segmentation model — SAM2

SAM2 (Segment Anything 2, from Meta) is a foundation model that can
draw a precise outline around any object you point at, without ever
having been trained specifically on shirts. We use it because the
alternative would be training our own garment-segmentation model on a
custom dataset, which is slow, narrow, and tends to fail on unusual
clothing.

In practice the user clicks on the garment in the browser. The click
travels to the backend, SAM2 figures out the mask, and the green
overlay you see in the UI is that mask painted on top of the photo.
Shift-clicking adds **negative** points to remove regions that were
included by mistake.

### Image preprocessing (filters before NST)

Before we hand the images to NST we run them through a short
image-processing pipeline. Each step has a small, defined purpose:

| Step | Why we do it |
|---|---|
| **Lanczos resize** to a short-side of 768 pixels | NST is expensive; we run it at a fixed manageable size and then scale the result back later. |
| **RGB → LAB** colour conversion | LAB separates lightness from colour, which lets us touch only one at a time. |
| **Histogram equalisation** on the L channel | Evens out lighting so a shadowy photo and a bright one start from the same base. |
| **Mild Gaussian blur** (σ ≈ 0.5) | Removes a tiny bit of camera sensor noise so it doesn't get misinterpreted as texture. |
| **Pattern suppression** (median + bilateral, optional, target garment only) | If the original target shirt already has a print we don't want, this flattens it before NST starts. (Off by default.) |
| **Mask edge softening** (Gaussian blur on mask) | Gives the final composite a soft edge instead of a hard cutout look. |

### Image postprocessing (filters after NST)

After NST finishes, the raw output is sharp but a bit speckly and its
colours can be off. We clean it up:

| Step | Why we do it |
|---|---|
| **3×3 median filter** | Removes salt-and-pepper specks the optimiser leaves behind. |
| **Mild bilateral filter** | Smooths small noise without softening real edges. |
| **LAB histogram match to the source** (blended via the *Colour strength* slider) | Optionally remaps the stylised image's colour distribution toward the source's actual fabric palette. |
| **YIQ luminance swap** | Replaces the brightness channel of the output with the brightness of the original target photo. This is what keeps the folds, shadows, and drape of the original garment visible through the new fabric. |
| **Laplacian sharpen** (k = 0.3) | Restores a little crispness lost to the previous smoothing steps. |

### Composite

Even after all of the above, the stylised image still includes the
person's face, hair, background — and those have drifted slightly
during NST. We don't want that. The composite step **alpha-blends** the
stylised garment region back over the **original** target photo, using
a softened version of the garment mask. Everything outside the mask is
the original pixels; everything inside the mask is the stylised result.

---

## What the refinements branch adds

The refinements branch keeps everything above and layers on extra
machinery from the Gatys 2017 follow-up paper, because the baseline ran
into specific problems with patterns.

### The pattern problem

NST as originally formulated transfers **texture statistics**, not
**specific shapes**. The Gram matrix is a global summary of "which
features co-occur" — it doesn't know that the source has hexagons or a
plaid grid. As long as the pattern is small enough to fit inside the
network's "viewing window" (the receptive field) at a deep layer, this
works — denim weave, fine prints, brush strokes all transfer fine. But
once a single repeating tile of the pattern is *larger* than that
window (e.g. ~100-pixel-tall hexagons at our 768-pixel working
resolution), the network sees one big blob at a time and the Gram
matrix flattens the pattern into a colour smear rather than reproducing
the motif.

### Two-stage NST (Gatys 2017 §6.2)

The follow-up paper's fix is straightforward: run NST first at a
**lower resolution**, where the same pattern now *does* fit inside the
viewing window, capture the macro arrangement, then upscale that
intermediate result and use it as the **starting point** for a second
NST pass at full resolution. The second pass refines the fine details
without disturbing the macro layout already encoded in the
initialisation. This is the heart of the refinements branch.

### The coarse → fine split

Coarse pass:
- Both images are resized to a 384-pixel short side.
- NST runs from random noise.
- Captures the source's macro pattern (hexagons, plaid blocks, large
  paisley).

Bridge:
- The coarse result is upscaled (bicubic) to 768-pixel short side.

Fine pass:
- Both images are freshly preprocessed at 768-pixel short side.
- NST runs **initialised from the upscaled coarse result** (instead
  of noise).
- Refines local detail, edges, and colour.

The user controls how to split the total iteration budget between the
two passes through the new **Coarse / fine split** slider.

### Pre-NST pattern suppression (improved)

The baseline's "suppress target pattern" option was implemented with a
single bilateral filter. The bilateral filter is edge-preserving, so
high-contrast features (like the dark dots on a chambray shirt) were
treated as edges and *survived* the filter rather than being smoothed
away.

The refinements branch fixes this by running a **median filter first**
(which removes small detail regardless of how much contrast it has),
then a stronger bilateral pass on top (to smooth moderate textures
while still preserving big folds and the garment silhouette).

### Colour strength tuning

Re-enabling the LAB histogram match at full strength on top of the new
two-stage NST gave outputs whose colours were overpowering (electric
plaid on a person standing in a garden). We introduced a **Colour
strength slider** (0 % – 100 %) that blends the stylised image with the
histogram-matched version. 0 % means we leave NST's natural, muted
colours alone; 100 % means we force the output's colour distribution
to match the source exactly.

### Output resolution change

The refinements branch outputs the final PNG at the **fine NST
resolution** (e.g. 768 × 1152 for a 2:3 portrait) rather than upscaling
back to the input's native size. This is a deliberate trade-off chosen
for prototyping speed — a final render at native resolution would
simply mean raising `TARGET_SHORT_SIDE` and running once more.

---

## How to use

### Start the servers

Open two terminals.

Backend:
```
cd backend
.\.venv\Scripts\activate    # on Windows
uvicorn main:app --host 127.0.0.1 --port 8000
```

Frontend:
```
cd frontend
npm run dev
```

Open `http://localhost:5173` in a browser.

### Step 1 — Pick & segment the target garment

The target is the photo of the person wearing the shirt you want to
re-style. In the **left** canvas:

1. Click **Choose File** and select the target photo.
2. Once the image appears, **click on the garment** (e.g. somewhere in
   the middle of the shirt). A green overlay appears — that is your
   mask, drawn by SAM2 around the garment.
3. If the green mask **includes things it shouldn't** (a bit of skin
   visible through a button gap, a necklace, a watch), **Shift+click**
   on those areas to remove them from the mask. The mask redraws
   after every click.
4. If the green mask **misses parts of the garment**, click on those
   missing parts (without Shift) to add them.
5. Counter under the canvas shows how many positive and negative
   points you've used so far.
6. When the green outline matches the garment, click **Use mask**.

### Step 2 — Pick & segment the source pattern

The source is the photo whose pattern you want to copy. Same flow on
the **right** canvas: upload, click on the patterned fabric, refine
with Shift+clicks, confirm with **Use mask**.

### Step 3 — Adjust sliders

Each slider, what it controls, and what changing it does **behind the
scenes**:

#### Style ↔ Content (α/β ratio)

- **What it controls**: how strongly to impose the source's texture
  statistics versus how much to respect the original target garment's
  structure.
- **Behind the scenes**: `α` is fixed at 1; `β = 1 / ratio`. The loss
  the optimiser minimises is `α · L_content + β · L_style`. Moving the
  slider left (smaller ratio) makes `β` larger, so the style term
  dominates and the optimiser pushes the result toward the source's
  Gram statistics. Moving the slider right makes `β` smaller, so the
  content term dominates and the original target's shape comes through
  more.
- **Rule of thumb**: `1e-4` is the paper-aligned default. `1e-5` is
  the most style-heavy the API allows.

#### Iterations (total)

- **What it controls**: total number of L-BFGS optimisation steps.
- **Behind the scenes**: each step recomputes the content and style
  losses, backpropagates through VGG-19, and adjusts the generated
  image's pixels. More steps means the optimiser gets closer to its
  best answer.
- **Rule of thumb**: 500 is the default. Below 300 the result tends
  to look unfinished; above 800 the gain becomes marginal.

#### Coarse / fine split *(refinements only)*

- **What it controls**: how the total iteration budget is divided
  between the 384-pixel coarse pass and the 768-pixel fine pass.
- **Behind the scenes**: the slider's value sets `coarse_fraction`
  between 0.1 and 0.9. The backend computes `coarse_iters =
  iters × coarse_fraction` and `fine_iters = iters − coarse_iters`.
  Slider **left** = more iterations to the fine pass (more detail and
  colour refinement). Slider **right** = more iterations to the coarse
  pass (better at recovering macro patterns like hexagons or plaid).
- **Rule of thumb**: default is 40 % coarse / 60 % fine. Bump coarse
  higher when the source has a large repeating motif you really want
  to see come through.

#### Colour strength *(refinements only)*

- **What it controls**: how strongly the stylised output's colour
  distribution is forced to match the source garment's colour
  distribution.
- **Behind the scenes**: at 0 %, the LAB histogram-match step is
  skipped entirely and you see NST's natural (more muted) colours. At
  100 %, the stylised image's LAB histogram is fully remapped to the
  source's. In between, the two are blended linearly.
- **Rule of thumb**: 50 % is the default. Lower if you want a softer,
  more painterly look; higher if you want the fabric's exact saturated
  colour palette.

#### Suppress target pattern (checkbox)

- **What it controls**: whether to flatten the original target
  garment's print before NST sees it.
- **Behind the scenes**: when ON, the target garment region (and only
  that region — the mask is used to bound the effect) is first run
  through a 9×9 **median filter** which removes small high-contrast
  detail like polka dots, then through a **bilateral filter** with a
  large colour sigma which smooths moderate textures while preserving
  the big folds and silhouette of the shirt. The rest of the image is
  untouched.
- **Rule of thumb**: turn it on when the target shirt has a visible
  pattern of its own and you don't want it bleeding through into the
  final result. Leave it off for plain shirts.

### Step 4 — Stylize and download

Click **Stylize**. The progress bar shows:

- The current **stage** of the pipeline (`preprocess`, then `nst`,
  then `postprocess`, then `composite`).
- During the `nst` stage, an iteration counter (e.g. `200 / 500`)
  counting across **both** the coarse and fine passes cumulatively.
- A percentage progress estimate.

When the stage label disappears and the percentage reaches 100, the
result image and a **Download PNG** button appear at the bottom of the
page.

---

## Citations

[1] Gatys, L. A., Ecker, A. S., Bethge, M. — *Image Style Transfer
Using Convolutional Neural Networks*, CVPR 2016.
See: `gatys-2016-neural-style-transfer.pdf` in the repo root.

[2] Gatys, L. A., Ecker, A. S., Bethge, M., Hertzmann, A., Shechtman,
E. — *Controlling Perceptual Factors in Neural Style Transfer*, CVPR
2017.
See: `gatys-2017-controlling-perceptual-factors.pdf` in the repo root.
