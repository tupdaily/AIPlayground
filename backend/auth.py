"""Supabase JWT verification for FastAPI."""

from fastapi import Depends, HTTPException, Request
from supabase import create_client
from config import settings


def get_current_user_id(request: Request) -> str:
    """Extract and verify user_id from Supabase JWT in Authorization header.

    Usage: user_id: str = Depends(get_current_user_id)
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing authorization token")

    token = auth_header.removeprefix("Bearer ")

    if not settings.supabase_url or not settings.supabase_service_role_key:
        raise HTTPException(status_code=500, detail="Supabase not configured")

    supabase = create_client(settings.supabase_url, settings.supabase_service_role_key)
    try:
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
        return user_response.user.id
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
