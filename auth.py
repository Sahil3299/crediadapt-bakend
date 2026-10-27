"""Password, JWT, and FastAPI authorization helpers."""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from typing import Annotated, Literal

import bcrypt
import jwt
from bson import ObjectId
from fastapi import Depends, HTTPException, Request, status
from jwt import InvalidTokenError
from pydantic import BaseModel

from db import get_database


UserRole = Literal["applicant", "officer"]
ACCESS_TOKEN_MINUTES = int(os.getenv("ACCESS_TOKEN_MINUTES", "20"))
REFRESH_TOKEN_DAYS = int(os.getenv("REFRESH_TOKEN_DAYS", "7"))
ALGORITHM = "HS256"


class CurrentUser(BaseModel):
    id: str
    email: str
    full_name: str
    role: UserRole


def _secret(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} is not configured.",
        )
    if len(value) < 32:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"{name} must be at least 32 characters.",
        )
    return value


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _make_token(user: dict, expires_delta: timedelta, token_type: str, secret_name: str) -> str:
    now = datetime.now(UTC)
    payload = {
        "sub": str(user["_id"]),
        "role": user["role"],
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
    }
    return jwt.encode(payload, _secret(secret_name), algorithm=ALGORITHM)


def create_access_token(user: dict) -> str:
    return _make_token(user, timedelta(minutes=ACCESS_TOKEN_MINUTES), "access", "JWT_SECRET")


def create_refresh_token(user: dict) -> str:
    return _make_token(user, timedelta(days=REFRESH_TOKEN_DAYS), "refresh", "JWT_REFRESH_SECRET")


def decode_token(token: str, expected_type: str) -> dict:
    secret_name = "JWT_SECRET" if expected_type == "access" else "JWT_REFRESH_SECRET"
    try:
        payload = jwt.decode(token, _secret(secret_name), algorithms=[ALGORITHM])
    except InvalidTokenError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token.") from exc
    if payload.get("type") != expected_type or not payload.get("sub") or not payload.get("role"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload.")
    return payload


def get_current_user(request: Request) -> CurrentUser:
    authorization = request.headers.get("Authorization", "")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_token(token, "access")
    try:
        user = get_database().users.find_one({"_id": ObjectId(payload["sub"]), "is_active": True})
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token subject.") from exc
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User account is unavailable.")
    if user["role"] != payload["role"]:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token role is no longer valid.")
    return CurrentUser(
        id=str(user["_id"]),
        email=user["email"],
        full_name=user["full_name"],
        role=user["role"],
    )


def require_role(role: UserRole):
    def dependency(current_user: Annotated[CurrentUser, Depends(get_current_user)]) -> CurrentUser:
        if current_user.role != role:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="You do not have access to this resource.")
        return current_user

    return dependency
