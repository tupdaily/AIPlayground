"""Google Cloud Storage operations for dataset files."""

from google.cloud import storage
from datetime import timedelta
from config import settings


def get_gcs_client() -> storage.Client:
    return storage.Client()


def get_bucket() -> storage.Bucket:
    client = get_gcs_client()
    return client.bucket(settings.gcs_bucket_name)


def upload_to_gcs(data: bytes, gcs_path: str, content_type: str = "application/octet-stream") -> str:
    """Upload bytes to GCS. Returns the gcs_path."""
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)
    blob.upload_from_string(data, content_type=content_type)
    return gcs_path


def generate_signed_url(gcs_path: str, expiration_hours: int = 1) -> str:
    """Generate a time-limited signed URL for downloading from GCS."""
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)
    url = blob.generate_signed_url(
        version="v4",
        expiration=timedelta(hours=expiration_hours),
        method="GET",
    )
    return url


def delete_from_gcs(gcs_path: str) -> None:
    """Delete an object from GCS."""
    bucket = get_bucket()
    blob = bucket.blob(gcs_path)
    blob.delete()
