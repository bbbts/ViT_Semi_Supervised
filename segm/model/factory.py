from pathlib import Path
import yaml
import torch
import os
import torch.nn as nn

from timm.models.helpers import load_pretrained, load_custom_pretrained
from timm.models.vision_transformer import default_cfgs
from timm.models.registry import register_model
from timm.models.vision_transformer import _create_vision_transformer

from segm.model.vit import VisionTransformer
from segm.model.utils import checkpoint_filter_fn
from segm.model.decoder import DecoderLinear
from segm.model.decoder import MaskTransformer
from segm.model.segmenter import Segmenter
import segm.utils.torch as ptu


# -----------------------------
# Register ViT backbone
# -----------------------------
@register_model
def vit_base_patch8_384(pretrained=False, **kwargs):
    """ViT-Base model (ViT-B/8)."""
    model_kwargs = dict(patch_size=8, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "vit_base_patch8_384",
        pretrained=pretrained,
        default_cfg=dict(
            url="",
            input_size=(3, 384, 384),
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
            num_classes=1000,
        ),
        **model_kwargs,
    )
    return model


# -----------------------------
# Create encoder (ViT)
# -----------------------------
def create_vit(model_cfg):
    model_cfg = model_cfg.copy()
    backbone = model_cfg.pop("backbone")
    normalization = model_cfg.pop("normalization")
    model_cfg["n_cls"] = 1000
    model_cfg["d_ff"] = 4 * model_cfg["d_model"]

    if backbone in default_cfgs:
        default_cfg = default_cfgs[backbone]
    else:
        default_cfg = dict(
            pretrained=False,
            num_classes=1000,
            drop_rate=0.0,
            drop_path_rate=0.0,
            drop_block_rate=None,
        )

    default_cfg["input_size"] = (
        3,
        model_cfg["image_size"][0],
        model_cfg["image_size"][1],
    )

    model = VisionTransformer(**model_cfg)

    if backbone == "vit_base_patch8_384":
        path = os.path.expandvars("$TORCH_HOME/hub/checkpoints/vit_base_patch8_384.pth")
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            filtered_dict = checkpoint_filter_fn(state_dict, model)
            model.load_state_dict(filtered_dict, strict=True)
    elif "deit" in backbone:
        load_pretrained(model, default_cfg, filter_fn=checkpoint_filter_fn)
    else:
        load_custom_pretrained(model, default_cfg)

    return model


# -----------------------------
# Create decoder
# -----------------------------
def create_decoder(encoder, decoder_cfg):
    decoder_cfg = decoder_cfg.copy()
    name = decoder_cfg.pop("name")
    decoder_cfg["d_encoder"] = encoder.d_model
    decoder_cfg["patch_size"] = encoder.patch_size

    if "linear" in name:
        decoder = DecoderLinear(**decoder_cfg)
    elif name == "mask_transformer":
        dim = encoder.d_model
        n_heads = dim // 64
        decoder_cfg["n_heads"] = n_heads
        decoder_cfg["d_model"] = dim
        decoder_cfg["d_ff"] = 4 * dim
        decoder = MaskTransformer(**decoder_cfg)
    else:
        raise ValueError(f"Unknown decoder: {name}")

    return decoder


# -----------------------------
# Create full segmenter
# -----------------------------
def create_segmenter(model_cfg):
    model_cfg = model_cfg.copy()
    decoder_cfg = model_cfg.pop("decoder")
    decoder_cfg["n_cls"] = model_cfg["n_cls"]

    encoder = create_vit(model_cfg)
    decoder = create_decoder(encoder, decoder_cfg)
    model = Segmenter(encoder, decoder, n_cls=model_cfg["n_cls"])
    return model


# -----------------------------
# Custom YAML Loader (fix for tuples and unknown Python objects)
# -----------------------------
class SafeLoaderWithPython(yaml.SafeLoader):
    pass


def construct_python_tuple(loader, node):
    return tuple(loader.construct_sequence(node))


def ignore_unknown(loader, tag_suffix, node):
    # Gracefully ignore any unknown Python object (like torchvision transforms)
    return None


SafeLoaderWithPython.add_constructor(
    "tag:yaml.org,2002:python/tuple", construct_python_tuple
)

SafeLoaderWithPython.add_multi_constructor("tag:yaml.org,2002:python/object:", ignore_unknown)


# -----------------------------
# Model loader
# -----------------------------
def load_model(model_path):
    variant_path = Path(model_path).parent / "variant.yml"

    with open(variant_path, "r") as f:
        text = f.read()

    variant = yaml.load(text, Loader=SafeLoaderWithPython)
    net_kwargs = variant["net_kwargs"]

    model = create_segmenter(net_kwargs)

    data = torch.load(model_path, map_location=ptu.device)
    checkpoint = data["model"]
    model.load_state_dict(checkpoint, strict=True)

    return model, variant
