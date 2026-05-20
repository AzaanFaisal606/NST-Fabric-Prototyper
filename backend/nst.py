from typing import Callable, Optional
import torch
import torch.nn.functional as F
from vgg import extract_features, STYLE_LAYERS, CONTENT_LAYER

# CONTENT layer (paper §3, Fig 5)
# STYLE layers (paper §3, equal weights w_l = 1/5)


# Gram matrix = the network's "summary" of which features fire together at a layer
def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    flat = feat.view(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


# masked Gram — same idea, but only counting pixels inside the garment region
def masked_gram(feat: torch.Tensor, mask_feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    m = mask_feat.view(b, 1, h * w)
    flat = feat.view(b, c, h * w) * m
    norm = m.sum().clamp(min=1.0) * c
    return flat @ flat.transpose(1, 2) / norm


def stylize(
    content_t: torch.Tensor,
    style_t: torch.Tensor,
    vgg: torch.nn.Sequential,
    alpha: float,
    beta: float,
    iters: int,
    progress_cb: Optional[Callable[[int], None]] = None,
    init_noise: bool = True,
    content_mask: Optional[torch.Tensor] = None,
    style_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    init_noise: start G from noise instead of content (option A).
    content_mask: [1,1,H,W] in [0,1] — 1 = target garment region, 0 = background.
                  Weights content loss and masks G's Gram so only garment is constrained.
    style_mask:   [1,1,H,W] in [0,1] over the style image — restricts source Gram targets
                  to the source garment region only.
    """
    content_target = extract_features(content_t, vgg, [CONTENT_LAYER])[CONTENT_LAYER].detach()

    # style targets — what the source's texture statistics look like at each layer (masked to garment)
    style_feats = extract_features(style_t, vgg, STYLE_LAYERS)
    style_targets: dict[str, torch.Tensor] = {}
    for l in STYLE_LAYERS:
        if style_mask is not None:
            _, _, fh, fw = style_feats[l].shape
            sm_l = F.interpolate(
                style_mask, size=(fh, fw), mode="bilinear", align_corners=False
            ).to(style_t.device)
            style_targets[l] = masked_gram(style_feats[l], sm_l).detach()
        else:
            style_targets[l] = gram_matrix(style_feats[l]).detach()

    if init_noise:
        G = torch.randn_like(content_t).mul(0.01).requires_grad_(True)
    else:
        G = content_t.clone().requires_grad_(True)

    optimizer = torch.optim.LBFGS([G], lr=1.0, max_iter=1)

    w_l = 1.0 / len(STYLE_LAYERS)

    # downsample mask to match conv4_2 feature spatial size if provided
    mask_feat: Optional[torch.Tensor] = None
    if content_mask is not None:
        # feature map size is not known until first forward pass; we'll resize lazily
        mask_feat = content_mask

    # cache per-style-layer resized content masks (lazy on first closure pass)
    style_mask_cache: dict[str, torch.Tensor] = {}
    step = {"i": 0, "mask_ready": False, "mask_resized": None}

    def closure():
        optimizer.zero_grad()

        feats = extract_features(G, vgg, STYLE_LAYERS + [CONTENT_LAYER])

        G_feat = feats[CONTENT_LAYER]

        if content_mask is not None and not step["mask_ready"]:
            # resize mask to feature spatial dims once
            _, _, fh, fw = G_feat.shape
            step["mask_resized"] = F.interpolate(
                mask_feat, size=(fh, fw), mode="bilinear", align_corners=False
            ).to(G.device)
            step["mask_ready"] = True

        if step["mask_resized"] is not None:
            m = step["mask_resized"]
            diff = (G_feat - content_target) ** 2
            # weight squared diff by mask: target pixels contribute fully, background near-zero
            content_loss = (diff * m).mean()
        else:
            content_loss = F.mse_loss(G_feat, content_target)

        style_loss = torch.zeros((), device=G.device)
        for l in STYLE_LAYERS:
            if content_mask is not None:
                # G is target image: mask its style stats with content_mask so only target garment region is constrained to source's Gram
                if l not in style_mask_cache:
                    _, _, fh, fw = feats[l].shape
                    style_mask_cache[l] = F.interpolate(
                        content_mask, size=(fh, fw), mode="bilinear", align_corners=False
                    ).to(G.device)
                G_gram = masked_gram(feats[l], style_mask_cache[l])
            else:
                G_gram = gram_matrix(feats[l])
            style_loss = style_loss + w_l * F.mse_loss(G_gram, style_targets[l])

        loss = alpha * content_loss + beta * style_loss
        loss.backward()
        return loss

    for i in range(iters):
        optimizer.step(closure)
        step["i"] = i + 1
        if progress_cb is not None:
            progress_cb(step["i"])

    return G.detach()
