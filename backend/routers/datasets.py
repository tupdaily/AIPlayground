"""Dataset management endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, Form, HTTPException, Request, UploadFile
from supabase import create_client

from auth import get_current_user_id
from config import settings
from storage import upload_to_gcs, delete_from_gcs
from training.datasets import BUILTIN_DATASETS
from training.dataset_validation import validate_csv, validate_image_zip

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


def _get_supabase():
    """Get a Supabase client using the service role key."""
    return create_client(settings.supabase_url, settings.supabase_service_role_key)


@router.get("/")
async def list_datasets(request: Request):
    """List all available datasets (built-in + user's custom if authenticated)."""
    builtin = [
        {
            "id": dataset_id,
            "name": info["name"],
            "description": info["description"],
            "input_shape": list(info["input_shape"]),
            "num_classes": info["num_classes"],
            "is_builtin": True,
        }
        for dataset_id, info in BUILTIN_DATASETS.items()
    ]

    custom = []
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and settings.supabase_url and settings.supabase_service_role_key:
        try:
            user_id = get_current_user_id(request)
            sb = _get_supabase()
            result = sb.table("datasets").select("*").eq("user_id", user_id).order("updated_at", desc=True).execute()
            for row in result.data:
                custom.append({
                    "id": f"custom:{row['id']}",
                    "name": row["name"],
                    "description": row.get("description", ""),
                    "input_shape": row["input_shape"],
                    "num_classes": row["num_classes"],
                    "num_samples": row.get("num_samples", 0),
                    "format": row["format"],
                    "is_builtin": False,
                })
        except Exception:
            pass  # If auth fails, just return built-in only

    return builtin + custom


@router.post("/upload")
async def upload_dataset(
    file: UploadFile,
    name: str = Form(...),
    label_column: str = Form(None),
    user_id: str = Depends(get_current_user_id),
):
    """Upload a custom dataset file (CSV or zip).

    Args:
        file: The dataset file (.csv or .zip)
        name: User-provided dataset name
        label_column: For CSV files, which column contains labels (optional, defaults to last column)
        user_id: Extracted from JWT by auth dependency
    """
    if not settings.gcs_bucket_name:
        raise HTTPException(status_code=500, detail="Cloud storage not configured")

    # Check file extension
    filename = file.filename or ""
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("csv", "zip"):
        raise HTTPException(status_code=400, detail="Only .csv and .zip files are supported")

    # Read file bytes (check size)
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    file_bytes = await file.read()
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(file_bytes) / 1024 / 1024:.1f} MB). Max is {settings.max_upload_size_mb} MB.",
        )

    # Validate and extract metadata
    try:
        if ext == "csv":
            meta = validate_csv(file_bytes, label_column)
        else:
            meta = validate_image_zip(file_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Upload to GCS
    dataset_uuid = str(uuid.uuid4())
    gcs_path = f"datasets/{user_id}/{dataset_uuid}/{filename}"
    content_type = "text/csv" if ext == "csv" else "application/zip"

    try:
        upload_to_gcs(file_bytes, gcs_path, content_type)
    except Exception as e:
        logger.exception("GCS upload failed for user=%s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to upload file to storage")

    # Insert metadata into Supabase
    try:
        sb = _get_supabase()
        row = {
            "id": dataset_uuid,
            "user_id": user_id,
            "name": name,
            "format": meta.format,
            "gcs_path": gcs_path,
            "input_shape": list(meta.input_shape),
            "num_classes": meta.num_classes,
            "num_samples": meta.num_samples,
            "file_size_bytes": len(file_bytes),
            "class_names": meta.class_names,
            "label_column": meta.label_column,
        }
        sb.table("datasets").insert(row).execute()
    except Exception as e:
        # Cleanup GCS on Supabase failure
        try:
            delete_from_gcs(gcs_path)
        except Exception:
            pass
        logger.exception("Supabase insert failed for user=%s: %s", user_id, e)
        raise HTTPException(status_code=500, detail="Failed to save dataset metadata")

    return {
        "id": f"custom:{dataset_uuid}",
        "name": name,
        "description": "",
        "input_shape": list(meta.input_shape),
        "num_classes": meta.num_classes,
        "num_samples": meta.num_samples,
        "format": meta.format,
        "class_names": meta.class_names,
        "is_builtin": False,
    }


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user_id),
):
    """Delete a custom dataset."""
    # Strip custom: prefix if present
    raw_id = dataset_id.removeprefix("custom:")

    sb = _get_supabase()
    result = sb.table("datasets").select("*").eq("id", raw_id).single().execute()
    if not result.data:
        raise HTTPException(status_code=404, detail="Dataset not found")

    row = result.data
    if row["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="Not authorized to delete this dataset")

    # Delete from GCS
    try:
        delete_from_gcs(row["gcs_path"])
    except Exception as e:
        logger.warning("GCS delete failed for dataset=%s: %s", raw_id, e)

    # Delete from Supabase
    sb.table("datasets").delete().eq("id", raw_id).execute()

    return {"status": "deleted"}
