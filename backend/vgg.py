import torch
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

# VGG-19 module index map for the layers we need (paper §3)
LAYER_INDEX = {
    "conv1_1": 0,
    "conv2_1": 5,
    "conv3_1": 10,
    "conv4_1": 19,
    "conv4_2": 21,    # content layer
    "conv5_1": 28,
}

# style layers (multi-scale texture, paper §3)
STYLE_LAYERS = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]
# content layer (paper §3.2)
CONTENT_LAYER = "conv4_2"


def load_vgg(device: torch.device) -> nn.Sequential:
    # load pretrained VGG-19 features, swap max-pool for avg-pool (paper §2), freeze
    features = vgg19(weights=VGG19_Weights.DEFAULT).features
    swapped = nn.Sequential()
    for i, layer in enumerate(features):
        if isinstance(layer, nn.MaxPool2d):
            swapped.add_module(str(i), nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            swapped.add_module(str(i), layer)
    swapped.eval().to(device)
    for p in swapped.parameters():
        p.requires_grad_(False)
    return swapped


def extract_features(x: torch.Tensor, vgg: nn.Sequential, layer_names: list[str]) -> dict[str, torch.Tensor]:
    # forward pass, capture activations at requested named layers
    wanted = {LAYER_INDEX[n]: n for n in layer_names}
    feats: dict[str, torch.Tensor] = {}
    h = x
    last = max(wanted.keys())
    for i, layer in enumerate(vgg):
        h = layer(h)
        if i in wanted:
            feats[wanted[i]] = h
        if i >= last:
            break
    return feats
