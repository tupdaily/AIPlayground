"""Dataset management: built-in datasets via torchvision + custom uploaded datasets."""

from __future__ import annotations

import csv
import io
import os
import random
import tempfile
import zipfile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2 as T

# ---------------------------------------------------------------------------
# SSL: use certifi's CA bundle so torchvision dataset downloads work when the
# system trust store is missing (e.g. some macOS Python installs). Without this,
# MNIST/CIFAR-10 downloads can fail with CERTIFICATE_VERIFY_FAILED.
# ---------------------------------------------------------------------------
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

# Root directory for built-in dataset files (MNIST, Fashion-MNIST, CIFAR-10).
# Override with DATASET_DATA_DIR env var. We create this dir before load so
# downloads never fail due to a missing path.
DATA_ROOT = os.path.abspath(os.environ.get("DATASET_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data")))


# ---------------------------------------------------------------------------
# Augmentation: Gaussian noise transform used when the user enables "noise" in
# the Augment block. Applied to the training split only (see get_dataloaders).
# ---------------------------------------------------------------------------
class AddGaussianNoise:
    """Add Gaussian noise to a tensor (for augmentation). Expects tensor in [0, 1]."""

    def __init__(self, amount: float = 0.08):
        self.amount = amount

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x + self.amount * torch.randn_like(x)).clamp(0.0, 1.0)


# Built-in image datasets (torchvision). Custom datasets use the "custom:" prefix
# and are registered via register_custom_dataset before training.
BUILTIN_DATASETS = {
    "mnist": {
        "name": "MNIST",
        "description": "Handwritten digits, 28x28 grayscale",
        "input_shape": (1, 28, 28),
        "num_classes": 10,
    },
    "fashion_mnist": {
        "name": "Fashion-MNIST",
        "description": "Clothing items, 28x28 grayscale",
        "input_shape": (1, 28, 28),
        "num_classes": 10,
    },
    "cifar10": {
        "name": "CIFAR-10",
        "description": "32x32 color images, 10 classes",
        "input_shape": (3, 32, 32),
        "num_classes": 10,
    },
}

# Registry for custom datasets (populated before training starts)
_custom_dataset_cache: dict[str, dict] = {}


def register_custom_dataset(dataset_id: str, meta: dict, signed_url: str) -> None:
    """Register custom dataset metadata and download URL for use during training.

    Must be called before get_dataset_shape() or get_dataloaders() for custom datasets.
    """
    _custom_dataset_cache[dataset_id] = {**meta, "signed_url": signed_url}


def get_dataset_info(dataset_id: str) -> dict | None:
    if dataset_id.startswith("custom:"):
        return _custom_dataset_cache.get(dataset_id)
    return BUILTIN_DATASETS.get(dataset_id)


def get_dataset_shape(dataset_id: str) -> tuple[int, ...]:
    if dataset_id.startswith("custom:"):
        meta = _custom_dataset_cache.get(dataset_id)
        if meta is None:
            raise ValueError(f"Custom dataset {dataset_id} not registered. Call register_custom_dataset first.")
        return tuple(meta["input_shape"])
    info = BUILTIN_DATASETS.get(dataset_id)
    if info is None:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    return info["input_shape"]


def _build_augment_transforms(augment_list: list[dict] | None) -> list:
    """Build a list of torchvision transforms from the Augment block's config.

    The frontend sends a list of { id, enabled, params } (e.g. rotation, hflip,
    noise). We only add transforms for enabled entries. Used in get_dataloaders
    to augment the training split (validation data is left unaugmented).
    """
    if not augment_list:
        return []
    out: list = []
    for a in augment_list:
        if not isinstance(a, dict) or a.get("enabled") is False:
            continue
        aid = a.get("id")
        params = a.get("params") or {}
        if aid == "rotation":
            deg = float(params.get("degrees", 15))
            out.append(T.RandomRotation(degrees=(-deg, deg)))
        elif aid == "hflip":
            out.append(T.RandomHorizontalFlip(p=0.5))
        elif aid == "vflip":
            out.append(T.RandomVerticalFlip(p=0.5))
        elif aid == "brightness":
            factor = float(params.get("factor", 0.2))
            out.append(T.ColorJitter(brightness=(1 - factor, 1 + factor)))
        elif aid == "contrast":
            factor = float(params.get("factor", 0.2))
            out.append(T.ColorJitter(contrast=(1 - factor, 1 + factor)))
        elif aid == "saturation":
            factor = float(params.get("factor", 0.3))
            out.append(T.ColorJitter(saturation=(1 - factor, 1 + factor)))
        elif aid == "noise":
            amount = float(params.get("amount", 0.08))
            out.append(AddGaussianNoise(amount=amount))
        elif aid == "blur":
            radius = float(params.get("radius", 1))
            k = max(3, min(15, int(2 * radius + 1) | 1))  # odd 3–15
            sigma = (radius * 0.5, radius * 1.0)
            out.append(T.GaussianBlur(kernel_size=(k, k), sigma=sigma))
    return out


# ---------------------------------------------------------------------------
# Sample image API (for Augment block preview in the UI)
# ---------------------------------------------------------------------------
# Cache loaded datasets so we don't re-download on every sample request.
# Key: dataset_id (mnist, fashion_mnist, cifar10). Value: torchvision VisionDataset.
_dataset_sample_cache: dict[str, torchvision.datasets.VisionDataset] = {}


def generate_random_sample_png(dataset_id: str) -> bytes:
    """Generate one random image with the correct shape for the dataset (no download).

    Used as a fallback when get_dataset_sample_png fails (e.g. SSL or network).
    Returns PNG bytes with the same dimensions as the dataset so the Augment
    preview always shows something for mnist, fashion_mnist, cifar10.
    """
    info = BUILTIN_DATASETS.get(dataset_id)
    if not info:
        raise ValueError(f"Unknown dataset: {dataset_id}")
    shape = info["input_shape"]  # (C, H, W)
    # Random image in [0, 1] with same shape as the dataset
    img_tensor = torch.rand(shape)
    return _tensor_to_png_bytes(img_tensor)


def _tensor_to_png_bytes(img_tensor: torch.Tensor) -> bytes:
    """Convert a CxHxW tensor in [0, 1] to PNG bytes.

    Handles 1-channel (MNIST, Fashion-MNIST) and 3-channel (CIFAR-10) using
    numpy + PIL so all three datasets render correctly in the browser.
    """
    x = img_tensor.cpu().clamp(0.0, 1.0)
    if x.dim() == 2:
        x = x.unsqueeze(0)
    arr = (x.numpy() * 255).astype(np.uint8)
    if arr.shape[0] == 1:
        pil = Image.fromarray(arr[0], mode="L")
    else:
        pil = Image.fromarray(np.transpose(arr, (1, 2, 0)), mode="RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def get_dataset_sample_png(dataset_id: str, index: int | None = None) -> bytes:
    """Load one image from the built-in dataset and return as PNG bytes.

    Used by GET /api/datasets/{id}/sample for the Augment block's live preview.
    Uses a random index when index is None. Datasets are cached in memory after
    first load so repeated requests are fast. Raises on failure; the router
    falls back to generate_random_sample_png when this raises.
    """
    if dataset_id not in _dataset_sample_cache:
        _dataset_sample_cache[dataset_id] = load_dataset(dataset_id)
    dataset = _dataset_sample_cache[dataset_id]
    n = len(dataset)
    if n == 0:
        raise ValueError("Dataset is empty")
    if index is None:
        index = random.randint(0, n - 1)
    else:
        index = max(0, min(index, n - 1))
    img_tensor, _ = dataset[index]
    if img_tensor.dim() == 2:
        img_tensor = img_tensor.unsqueeze(0)
    return _tensor_to_png_bytes(img_tensor)


def load_dataset(dataset_id: str) -> torchvision.datasets.VisionDataset:
    """Load a built-in dataset with ToTensor only.

    Augmentations (from the Augment block) are applied in get_dataloaders, not here.
    Creates DATA_ROOT if needed so the first download never fails on path.
    """
    os.makedirs(DATA_ROOT, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    match dataset_id:
        case "mnist":
            return torchvision.datasets.MNIST(
                DATA_ROOT, train=True, download=True, transform=transform
            )
        case "fashion_mnist":
            return torchvision.datasets.FashionMNIST(
                DATA_ROOT, train=True, download=True, transform=transform
            )
        case "cifar10":
            return torchvision.datasets.CIFAR10(
                DATA_ROOT, train=True, download=True, transform=transform
            )
        case _:
            raise ValueError(f"Unknown dataset: {dataset_id}")


class _TransformSubset(Dataset):
    """Wraps a subset and applies an extra transform to (image, target) -> (transform(image), target)."""

    def __init__(self, subset: torch.utils.data.Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.subset)

    def __getitem__(self, idx: int):
        x, y = self.subset[idx]
        return self.transform(x), y


def get_dataloaders(
    dataset_id: str,
    batch_size: int = 64,
    train_split: float = 0.8,
    augment_config: list[dict] | None = None,
) -> tuple[DataLoader, DataLoader]:
    """Load dataset and split into train/validation DataLoaders.
    Data source is always the dataset_id (from the Input block). augment_config, when
    provided (from an Augment block connected to Input), applies image transforms to the
    training split only; used for image datasets (e.g. mnist, fashion_mnist, cifar10).
    """
    if dataset_id.startswith("custom:"):
        return _get_custom_dataloaders(dataset_id, batch_size, train_split)

    dataset = load_dataset(dataset_id)

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_subset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Apply Augment block config (rotation, flip, noise, etc.) to training data only.
    augment_transforms = _build_augment_transforms(augment_config)
    if augment_transforms:
        train_dataset = _TransformSubset(
            train_subset, transforms.Compose([*augment_transforms])
        )
    else:
        train_dataset = train_subset

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Custom dataset support
# ---------------------------------------------------------------------------


class CSVDataset(Dataset):
    """PyTorch Dataset for CSV files with numeric features and a label column."""

    def __init__(self, csv_path: str, label_column: str, input_shape: tuple[int, ...]):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = [h.strip() for h in next(reader)]
            label_idx = headers.index(label_column)
            feature_idxs = [i for i in range(len(headers)) if i != label_idx]

            features = []
            labels_raw = []
            for row in reader:
                if not any(cell.strip() for cell in row):
                    continue
                features.append([float(row[i]) for i in feature_idxs])
                labels_raw.append(row[label_idx].strip())

        # Map string labels to integer indices
        unique_labels = sorted(set(labels_raw))
        label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}

        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor([label_to_idx[lbl] for lbl in labels_raw], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def _download_to_temp(signed_url: str, filename: str) -> str:
    """Download a file from a signed URL to a temp directory. Returns the file path."""
    import requests

    temp_dir = tempfile.mkdtemp(prefix="aiplayground_dataset_")
    filepath = os.path.join(temp_dir, filename)
    response = requests.get(signed_url, timeout=300)
    response.raise_for_status()
    with open(filepath, "wb") as f:
        f.write(response.content)
    return filepath


def _get_custom_dataloaders(
    dataset_id: str,
    batch_size: int,
    train_split: float,
) -> tuple[DataLoader, DataLoader]:
    """Load a custom dataset from GCS via signed URL and return DataLoaders."""
    meta = _custom_dataset_cache.get(dataset_id)
    if meta is None:
        raise ValueError(f"Custom dataset {dataset_id} not registered")

    signed_url = meta["signed_url"]
    fmt = meta["format"]
    input_shape = tuple(meta["input_shape"])

    if fmt == "csv":
        filepath = _download_to_temp(signed_url, "data.csv")
        dataset = CSVDataset(filepath, meta["label_column"], input_shape)
    elif fmt == "image_folder":
        zip_path = _download_to_temp(signed_url, "data.zip")
        extract_dir = os.path.join(os.path.dirname(zip_path), "images")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)

        # Find the root that contains class folders
        # Could be extract_dir directly or extract_dir/some_root_folder
        image_root = _find_image_folder_root(extract_dir)

        target_h, target_w = input_shape[1], input_shape[2]
        transform = transforms.Compose([
            transforms.Resize((target_h, target_w)),
            transforms.ToTensor(),
        ])
        dataset = torchvision.datasets.ImageFolder(image_root, transform=transform)
    else:
        raise ValueError(f"Unknown custom dataset format: {fmt}")

    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    return train_loader, val_loader


def _find_image_folder_root(extract_dir: str) -> str:
    """Find the directory containing class subdirectories.

    Handles both:
      extract_dir/class_a/img.jpg    -> returns extract_dir
      extract_dir/root_dir/class_a/img.jpg -> returns extract_dir/root_dir
    """
    entries = [
        e for e in os.listdir(extract_dir)
        if not e.startswith(".") and e != "__MACOSX"
    ]

    # Check if entries are directories containing images (i.e., this is the root)
    has_image_subdirs = any(
        os.path.isdir(os.path.join(extract_dir, e)) and
        any(
            f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"))
            for f in os.listdir(os.path.join(extract_dir, e))
            if os.path.isfile(os.path.join(extract_dir, e, f))
        )
        for e in entries
    )

    if has_image_subdirs:
        return extract_dir

    # Maybe there's a single wrapper directory
    if len(entries) == 1 and os.path.isdir(os.path.join(extract_dir, entries[0])):
        return os.path.join(extract_dir, entries[0])

    return extract_dir
