"""MongoDB connection and index management for CrediAdapt.

The application deliberately keeps the Mongo client process-wide.  PyMongo's
client is thread-safe and pools connections, which is a better fit for the
sync FastAPI endpoints than opening a connection per request.
"""

from __future__ import annotations

import os
from urllib.parse import urlparse

from pymongo import ASCENDING, MongoClient
from pymongo.database import Database


_client: MongoClient | None = None
_database: Database | None = None


def get_database() -> Database:
    """Return the configured database or raise a safe configuration error."""
    global _client, _database
    if _database is not None:
        return _database

    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError("MONGODB_URI is not configured.")

    _client = MongoClient(uri, serverSelectionTimeoutMS=5000, connectTimeoutMS=5000)
    configured_name = os.getenv("MONGODB_DB_NAME")
    if configured_name:
        _database = _client[configured_name]
    else:
        parsed = urlparse(uri)
        database_name = parsed.path.lstrip("/").split("/")[0]
        _database = _client[database_name or "crediadapt"]
    return _database


def ensure_indexes() -> None:
    """Create the indexes required by the lending workflow."""
    db = get_database()
    db.users.create_index([("email", ASCENDING)], unique=True, name="users_email_unique")
    db.applications.create_index([("applicant_id", ASCENDING)], name="applications_applicant")
    db.applications.create_index([("status", ASCENDING)], name="applications_status")
    db.audit_logs.create_index([("application_id", ASCENDING)], name="audit_application")


def close_database() -> None:
    global _client, _database
    if _client is not None:
        _client.close()
    _client = None
    _database = None
