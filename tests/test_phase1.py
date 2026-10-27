"""Phase 1 contract tests using an in-memory MongoDB-shaped test double.

Run after installing ``backend/requirements.txt``:
    python -m unittest discover -s backend/tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from bson import ObjectId
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
os.environ.setdefault("JWT_SECRET", "test-access-secret-that-is-at-least-32-bytes")
os.environ.setdefault("JWT_REFRESH_SECRET", "test-refresh-secret-that-is-at-least-32-bytes")

import api
import auth


class InsertResult:
    def __init__(self, inserted_id: ObjectId):
        self.inserted_id = inserted_id


class Cursor(list):
    def sort(self, _field: str, _direction: int):
        return self


class Collection:
    def __init__(self, unique_email: bool = False):
        self.documents: list[dict] = []
        self.unique_email = unique_email

    def create_index(self, *_args, **_kwargs):
        return "test-index"

    def insert_one(self, document: dict) -> InsertResult:
        if self.unique_email and any(item.get("email") == document.get("email") for item in self.documents):
            from pymongo.errors import DuplicateKeyError

            raise DuplicateKeyError("duplicate email")
        stored = dict(document)
        stored.setdefault("_id", ObjectId())
        self.documents.append(stored)
        return InsertResult(stored["_id"])

    def find_one(self, query: dict):
        for document in self.documents:
            if all(document.get(key) == value for key, value in query.items()):
                return document
        return None

    def find(self, query: dict):
        return Cursor([item for item in self.documents if all(item.get(key) == value for key, value in query.items())])


class Database:
    def __init__(self):
        self.users = Collection(unique_email=True)
        self.applications = Collection()
        self.audit_logs = Collection()
        self.counters = Collection()


class PhaseOneApiTests(unittest.TestCase):
    def setUp(self):
        self.database = Database()
        self.patches = [
            patch.object(api, "get_database", return_value=self.database),
            patch.object(auth, "get_database", return_value=self.database),
            patch.object(api, "ensure_indexes"),
            patch.object(api, "close_database"),
        ]
        for mocked in self.patches:
            mocked.start()
        self.client_context = TestClient(api.app)
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)
        for mocked in reversed(self.patches):
            mocked.stop()

    def register(self, email: str = "applicant@example.com"):
        return self.client.post(
            "/api/auth/register",
            json={"email": email, "password": "correct-horse", "full_name": "Test Applicant"},
        )

    def test_applicant_registration_and_authenticated_application_list(self):
        response = self.register()
        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["user"]["role"], "applicant")
        self.assertIn("crediadapt_refresh", response.headers.get("set-cookie", ""))

        applications = self.client.get(
            "/api/applications",
            headers={"Authorization": f"Bearer {payload['access_token']}"},
        )
        self.assertEqual(applications.status_code, 200)
        self.assertEqual(applications.json(), {"items": []})

    def test_protected_endpoint_rejects_missing_token(self):
        response = self.client.get("/api/applications")
        self.assertEqual(response.status_code, 401)

    def test_applicant_cannot_open_officer_queue(self):
        registration = self.register("queue-applicant@example.com").json()
        response = self.client.get(
            "/api/officer/queue",
            headers={"Authorization": f"Bearer {registration['access_token']}"},
        )
        self.assertEqual(response.status_code, 403)

    def test_login_rejects_role_mismatch(self):
        self.database.users.insert_one(
            {
                "email": "officer@example.com",
                "password_hash": auth.hash_password("correct-horse"),
                "full_name": "Test Officer",
                "role": "officer",
                "is_active": True,
            }
        )
        response = self.client.post(
            "/api/auth/login",
            json={"email": "officer@example.com", "password": "correct-horse", "role": "applicant"},
        )
        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
