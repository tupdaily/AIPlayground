"""Validation and metadata extraction for custom dataset uploads."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass


@dataclass
class DatasetMeta:
    format: str  # "csv" or "image_folder"
    input_shape: tuple[int, ...]
    num_classes: int
    num_samples: int
    class_names: list[str]
    label_column: str | None = None  # only for CSV


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def validate_csv(file_bytes: bytes, label_column: str | None = None) -> DatasetMeta:
    """Validate a CSV file and extract dataset metadata.

    Args:
        file_bytes: Raw CSV file content.
        label_column: Which column contains labels. If None, uses the last column.

    Returns:
        DatasetMeta with inferred shape, classes, etc.

    Raises:
        ValueError: If the CSV is malformed or unsuitable for training.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError("CSV file must be UTF-8 encoded")

    reader = csv.reader(io.StringIO(text))
    rows = list(reader)

    if len(rows) < 2:
        raise ValueError("CSV must have a header row and at least one data row")

    headers = [h.strip() for h in rows[0]]
    if not headers:
        raise ValueError("CSV header row is empty")

    # Determine label column
    if label_column is None:
        label_column = headers[-1]
    if label_column not in headers:
        raise ValueError(f"Label column '{label_column}' not found in CSV headers: {headers}")

    label_idx = headers.index(label_column)
    feature_idxs = [i for i in range(len(headers)) if i != label_idx]

    if not feature_idxs:
        raise ValueError("CSV must have at least one feature column besides the label column")

    # Parse data rows
    data_rows = rows[1:]
    data_rows = [r for r in data_rows if any(cell.strip() for cell in r)]  # skip blank rows

    if len(data_rows) < 10:
        raise ValueError(f"CSV has only {len(data_rows)} data rows; need at least 10")

    # Validate that feature columns are numeric
    for row_idx, row in enumerate(data_rows[:100]):  # check first 100 rows
        if len(row) != len(headers):
            raise ValueError(f"Row {row_idx + 2} has {len(row)} columns, expected {len(headers)}")
        for fi in feature_idxs:
            try:
                float(row[fi])
            except ValueError:
                raise ValueError(
                    f"Non-numeric value '{row[fi]}' in feature column '{headers[fi]}' at row {row_idx + 2}"
                )

    # Extract class info
    labels = [row[label_idx].strip() for row in data_rows]
    unique_labels = sorted(set(labels))

    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes, found {len(unique_labels)}")

    return DatasetMeta(
        format="csv",
        input_shape=(len(feature_idxs),),
        num_classes=len(unique_labels),
        num_samples=len(data_rows),
        class_names=unique_labels,
        label_column=label_column,
    )


def validate_image_zip(file_bytes: bytes) -> DatasetMeta:
    """Validate a zip file with ImageFolder structure and extract metadata.

    Expected structure:
        dataset.zip/
          class_a/
            img1.jpg
            img2.png
          class_b/
            img3.jpg

    Args:
        file_bytes: Raw zip file content.

    Returns:
        DatasetMeta with inferred shape, classes, etc.

    Raises:
        ValueError: If the zip structure is invalid.
    """
    try:
        zf = zipfile.ZipFile(io.BytesIO(file_bytes))
    except zipfile.BadZipFile:
        raise ValueError("File is not a valid zip archive")

    # Collect image paths grouped by class directory
    class_images: dict[str, list[str]] = {}

    for name in zf.namelist():
        # Skip directories, __MACOSX, hidden files
        if name.endswith("/") or "/__MACOSX" in name or "__MACOSX" in name:
            continue
        if any(part.startswith(".") for part in name.split("/")):
            continue

        parts = name.split("/")

        # Handle both "class/image.jpg" and "root_dir/class/image.jpg"
        # Find the image file and its parent directory (the class)
        ext = "." + parts[-1].rsplit(".", 1)[-1].lower() if "." in parts[-1] else ""
        if ext not in IMAGE_EXTENSIONS:
            continue

        if len(parts) == 2:
            # class/image.jpg
            class_name = parts[0]
        elif len(parts) == 3:
            # root_dir/class/image.jpg
            class_name = parts[1]
        else:
            continue  # skip deeply nested or flat files

        class_images.setdefault(class_name, []).append(name)

    if not class_images:
        raise ValueError(
            "No valid ImageFolder structure found. Expected: class_name/image.jpg "
            "or root_dir/class_name/image.jpg"
        )

    if len(class_images) < 2:
        raise ValueError(f"Need at least 2 class folders, found {len(class_images)}: {list(class_images.keys())}")

    # Validate minimum images per class
    for cls, imgs in class_images.items():
        if len(imgs) < 2:
            raise ValueError(f"Class '{cls}' has only {len(imgs)} image(s); need at least 2")

    # Sample one image to determine dimensions
    sample_path = next(iter(next(iter(class_images.values()))))
    try:
        from PIL import Image
        img_data = zf.read(sample_path)
        img = Image.open(io.BytesIO(img_data))
        width, height = img.size
        channels = len(img.getbands())
    except Exception as e:
        raise ValueError(f"Could not read sample image '{sample_path}': {e}")

    class_names = sorted(class_images.keys())
    total_images = sum(len(imgs) for imgs in class_images.values())

    return DatasetMeta(
        format="image_folder",
        input_shape=(channels, height, width),
        num_classes=len(class_names),
        num_samples=total_images,
        class_names=class_names,
    )
