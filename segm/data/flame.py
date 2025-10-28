# Path: /home/AD.UNLV.EDU/bhattb3/segmenter_semi_supervised_AGAIN/segm/data/flame.py

import os
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import yaml
from segm.data import utils

# -------------------------------------------------------------------
# Dataset configuration
# -------------------------------------------------------------------
FLAME_CATS_PATH = Path(__file__).parent / "config" / "flame.yml"
IGNORE_LABEL = 255

STATS = {
    "vit": {"mean": (0.5, 0.5, 0.5), "std": (0.5, 0.5, 0.5)},
    "default": {"mean": (127.5, 127.5, 127.5), "std": (127.5, 127.5, 127.5)},
}

# Mask pixel ? class ID mapping
PALETTE_TO_ID = {
    0: 0,  # background
    1: 1,  # fire
}

# -------------------------------------------------------------------
# Dataset class
# -------------------------------------------------------------------
class FlameDataset(Dataset):
    def __init__(self, image_size=512, crop_size=512, split="train",
                 normalization="vit", root=None, labeled_ratio=1.0, ssl=False):
        """
        Args:
            image_size: Resize dimension for images and masks
            crop_size:  (unused for now, kept for consistency)
            split:      "train", "val", or "test"
            normalization: which normalization to use ("vit" or "default")
            root:       dataset root path
            labeled_ratio: ratio of labeled samples for SSL
            ssl:        whether semi-supervised learning is active
        """
        self.root = Path(root or "/home/AD.UNLV.EDU/bhattb3/Datasets/Flame/")
        self.split = split
        self.ssl = ssl
        self.labeled_ratio = labeled_ratio

        self.image_dir = self.root / "images" / split
        self.mask_dir = self.root / "masks" / split

        self.images = sorted(list(self.image_dir.glob("*.jpg")) + list(self.image_dir.glob("*.png")))
        self.masks = sorted(list(self.mask_dir.glob("*.png")))

        if len(self.images) == 0 or (split != "test" and len(self.masks) == 0):
            raise RuntimeError(f"No images or masks found. Check paths: {self.image_dir}, {self.mask_dir}")
        if split != "test" and len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) and masks ({len(self.masks)}) do not match!")

        self.image_size = image_size
        self.crop_size = crop_size
        self.normalization = STATS.get(normalization, STATS["default"]).copy()

        # Transforms
        self.transform_img = T.Compose([
            T.Resize((self.image_size, self.image_size)),
            T.ToTensor(),
            T.Normalize(mean=self.normalization["mean"], std=self.normalization["std"]),
        ])
        self.transform_mask = T.Resize((self.image_size, self.image_size), interpolation=T.InterpolationMode.NEAREST)

        # Split labeled/unlabeled portion (for semi-supervised learning)
        if split == "train" and labeled_ratio < 1.0:
            combined = list(zip(self.images, self.masks))
            random.shuffle(combined)
            n_labeled = int(len(combined) * labeled_ratio)
            self.labeled_samples = combined[:n_labeled]
            self.unlabeled_samples = combined[n_labeled:]
        else:
            self.labeled_samples = list(zip(self.images, self.masks))
            self.unlabeled_samples = []

        # Class names and colors
        with open(FLAME_CATS_PATH, "r") as f:
            self.cat_desc = yaml.load(f, Loader=yaml.FullLoader)
        self.names, self.colors = utils.dataset_cat_description(FLAME_CATS_PATH)
        self.n_cls = len(self.cat_desc)
        self.ignore_label = IGNORE_LABEL

    # -------------------------------------------------------------------
    def __len__(self):
        if self.ssl and self.split == "train":
            return len(self.labeled_samples) + len(self.unlabeled_samples)
        return len(self.images)

    # -------------------------------------------------------------------
    def __getitem__(self, idx):
        if self.ssl and self.split == "train":
            if idx < len(self.labeled_samples):
                img_path, mask_path = self.labeled_samples[idx]
                is_labeled = True
            else:
                img_path, mask_path = self.unlabeled_samples[idx - len(self.labeled_samples)]
                is_labeled = False
        else:
            img_path = self.images[idx]
            mask_path = self.masks[idx] if self.split != "test" else None
            is_labeled = True if mask_path is not None else False

        # --- Load and preprocess image ---
        img = Image.open(img_path).convert("RGB")
        img = self.transform_img(img)

        # --- Load and preprocess mask ---
        mask = None
        if is_labeled:
            mask = Image.open(mask_path).convert("L")
            mask = self.transform_mask(mask)
            mask = np.array(mask, dtype=np.uint8)

            # Map pixel values ? class IDs
            mask_mapped = np.full_like(mask, IGNORE_LABEL)
            for pixel_val, class_id in PALETTE_TO_ID.items():
                mask_mapped[mask == pixel_val] = class_id

            # --- SAFETY PATCH: clamp out-of-bounds pixels ---
            mask_mapped = np.where((mask_mapped >= self.n_cls) & (mask_mapped != IGNORE_LABEL),
                                   IGNORE_LABEL, mask_mapped)

            mask = torch.from_numpy(mask_mapped).long()

        image_id = os.path.basename(img_path).split('.')[0]

        # --- Sanity check (only first batch) ---
        if idx == 0:
            print(f"[DEBUG] Image shape: {img.shape}, Mask shape: {mask.shape if mask is not None else 'None'}")

        return {
            "image": img,
            "mask": mask,
            "id": image_id,
            "is_labeled": is_labeled
        }

    # -------------------------------------------------------------------
    def get_gt_seg_maps(self):
        """Returns ground-truth segmentation maps as numpy arrays."""
        gt_seg_maps = {}
        for img_path, mask_path in zip(self.images, self.masks):
            mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
            mask_mapped = np.full_like(mask, IGNORE_LABEL)
            for pixel_val, class_id in PALETTE_TO_ID.items():
                mask_mapped[mask == pixel_val] = class_id

            # --- SAFETY PATCH ---
            mask_mapped = np.where((mask_mapped >= self.n_cls) & (mask_mapped != IGNORE_LABEL),
                                   IGNORE_LABEL, mask_mapped)

            gt_seg_maps[os.path.basename(img_path).split('.')[0]] = mask_mapped
        return gt_seg_maps
