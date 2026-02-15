"""Dataset management: built-in datasets via torchvision + custom uploaded datasets."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import zipfile

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms


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


def load_dataset(dataset_id: str) -> torchvision.datasets.VisionDataset:
    """Load a built-in dataset."""
    transform = transforms.Compose([transforms.ToTensor()])

    match dataset_id:
        case "mnist":
            return torchvision.datasets.MNIST(
                "./data", train=True, download=True, transform=transform
            )
        case "fashion_mnist":
            return torchvision.datasets.FashionMNIST(
                "./data", train=True, download=True, transform=transform
            )
        case "cifar10":
            return torchvision.datasets.CIFAR10(
                "./data", train=True, download=True, transform=transform
            )
        case _:
            raise ValueError(f"Unknown dataset: {dataset_id}")


def get_dataloaders(
    dataset_id: str,
    batch_size: int = 64,
    train_split: float = 0.8,
) -> tuple[DataLoader, DataLoader]:
    """Load dataset and split into train/validation DataLoaders."""
    if dataset_id.startswith("custom:"):
        return _get_custom_dataloaders(dataset_id, batch_size, train_split)

    dataset = load_dataset(dataset_id)

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
