from typing import Callable, Optional
import torch
import torch.nn.functional as F
from vgg import extract_features, STYLE_LAYERS, CONTENT_LAYER


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    # feature correlations across channels, normalized (paper eq. 3 / our normalization)
    b, c, h, w = feat.shape
    flat = feat.view(b, c, h * w)
    return flat @ flat.transpose(1, 2) / (c * h * w)


def stylize(
    content_t: torch.Tensor,
    style_t: torch.Tensor,
    vgg: torch.nn.Sequential,
    alpha: float,
    beta: float,
    iters: int,
    progress_cb: Optional[Callable[[int], None]] = None,
) -> torch.Tensor:
    # content features (target for L_content)
    content_target = extract_features(content_t, vgg, [CONTENT_LAYER])[CONTENT_LAYER].detach()

    # style Gram matrices (targets for L_style)
    style_feats = extract_features(style_t, vgg, STYLE_LAYERS)
    style_targets = {l: gram_matrix(style_feats[l]).detach() for l in STYLE_LAYERS}

    # init G as content image (paper Fig 6 - deterministic, biased to letter shape)
    G = content_t.clone().requires_grad_(True)

    # L-BFGS optimizer on pixels of G (paper §3)
    optimizer = torch.optim.LBFGS([G], lr=1.0, max_iter=1)

    # equal weights across the 5 style layers (paper §3)
    w_l = 1.0 / len(STYLE_LAYERS)

    step = {"i": 0}

    def closure():
        # closure runs each L-BFGS iteration
        optimizer.zero_grad()

        # extract G features at all layers we need
        feats = extract_features(G, vgg, STYLE_LAYERS + [CONTENT_LAYER])

        # content loss (MSE at conv4_2)
        content_loss = F.mse_loss(feats[CONTENT_LAYER], content_target)

        # style loss = sum_l w_l * MSE(gram(G), gram(style)) at each style layer
        style_loss = torch.zeros((), device=G.device)
        for l in STYLE_LAYERS:
            style_loss = style_loss + w_l * F.mse_loss(gram_matrix(feats[l]), style_targets[l])

        # total loss = α * content + β * style
        loss = alpha * content_loss + beta * style_loss
        loss.backward()
        return loss

    # main optimization loop
    for i in range(iters):
        optimizer.step(closure)
        step["i"] = i + 1
        if progress_cb is not None:
            progress_cb(step["i"])

    return G.detach()
