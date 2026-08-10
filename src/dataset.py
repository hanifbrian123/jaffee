"""JAFFE image dataset: grayscale .tiff -> 3-channel ImageNet-normalized tensor.

Mirrors the CASME II preprocessing discipline: one shared code path for train
augmentation and deterministic TTA, so a model is never evaluated with a recipe
different from what its ensemble partners saw.
"""
import os

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_gray(path, base_size):
    """Load a .tiff as grayscale and resize to base_size (no display)."""
    image = cv2.imread(os.path.join(HERE, path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"gagal membaca {path}")
    return cv2.resize(image, (base_size, base_size), interpolation=cv2.INTER_AREA)


def preload(samples, base_size):
    """key -> (base_size, base_size) uint8, kept in RAM (213 tiny images)."""
    return {s["key"]: load_gray(s["path"], base_size) for s in samples}


def _to_tensor(image):
    # image: (H,W,3) float32 in [0,1] -> normalized (3,H,W)
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.transpose(image, (2, 0, 1)).copy()).float()


def prepare(gray, cfg, train, view=None):
    """Grayscale (Hb,Wb) -> augmented/normalized (3,s,s) tensor."""
    view = view or {}
    size = cfg["img_size"]
    base = gray.shape[0]
    if train:
        # random crop within a small margin
        margin = max(0, base - size)
        top = np.random.randint(0, margin + 1) if margin else 0
        left = np.random.randint(0, margin + 1) if margin else 0
        crop = gray[top:top + size, left:left + size]
        if crop.shape != (size, size):
            crop = cv2.resize(crop, (size, size))
        angle = np.random.uniform(-cfg.get("rotate", 12), cfg.get("rotate", 12))
        matrix = cv2.getRotationMatrix2D((size / 2, size / 2), angle, 1.0)
        crop = cv2.warpAffine(crop, matrix, (size, size),
                              borderMode=cv2.BORDER_REFLECT_101)
        flip = cfg.get("hflip", True) and np.random.rand() < 0.5
    else:
        offset = (base - size) // 2
        if view.get("crop") == "top_left":
            offset_y = offset_x = 0
        elif view.get("crop") == "bottom_right":
            offset_y = offset_x = max(0, base - size)
        else:
            offset_y = offset_x = offset
        crop = gray[offset_y:offset_y + size, offset_x:offset_x + size]
        if crop.shape != (size, size):
            crop = cv2.resize(crop, (size, size))
        flip = bool(view.get("flip", False))

    image = crop.astype(np.float32) / 255.0
    if train:
        jitter = cfg.get("brightness", 0.0)
        if jitter:
            image = np.clip(image * (1.0 + np.random.uniform(-jitter, jitter)),
                            0, 1)
    image = np.repeat(image[:, :, None], 3, axis=2)          # gray -> 3 channel
    if flip:
        image = image[:, ::-1, :].copy()
    if train and cfg.get("random_erase", 0.0) > 0 \
            and np.random.rand() < cfg["random_erase"]:
        eh = np.random.randint(size // 8, size // 3)
        ew = np.random.randint(size // 8, size // 3)
        ey = np.random.randint(0, size - eh)
        ex = np.random.randint(0, size - ew)
        image[ey:ey + eh, ex:ex + ew, :] = 0.0
    return _to_tensor(image)


def tta_views(n):
    """Deterministic eval views (center, flip, corners)."""
    bank = [
        {"crop": "center", "flip": False},
        {"crop": "center", "flip": True},
        {"crop": "top_left", "flip": False},
        {"crop": "bottom_right", "flip": True},
    ]
    return bank[:max(1, n)]


class JaffeDataset(Dataset):
    def __init__(self, samples, arrays, cfg, train, view=None):
        self.samples = samples
        self.arrays = arrays
        self.cfg = cfg
        self.train = train
        self.view = view

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = prepare(self.arrays[s["key"]], self.cfg, self.train, self.view)
        return x, int(s["label"])
