"""Create an officer account outside the public HTTP surface.

Usage:
    python scripts/create_officer.py --email officer@example.com --name "Loan Officer"
"""

from __future__ import annotations

import argparse
import getpass
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from auth import hash_password
from db import ensure_indexes, get_database


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a CrediAdapt loan-officer account.")
    parser.add_argument("--email", required=True, help="Officer email address")
    parser.add_argument("--name", required=True, help="Officer full name")
    args = parser.parse_args()
    email = args.email.strip().lower()
    if "@" not in email or "." not in email.rsplit("@", 1)[-1]:
        parser.error("--email must be a valid email address")

    password = getpass.getpass("Password (at least 8 characters): ")
    confirm = getpass.getpass("Confirm password: ")
    if len(password) < 8:
        parser.error("Password must be at least 8 characters")
    if password != confirm:
        parser.error("Passwords do not match")

    ensure_indexes()
    try:
        get_database().users.insert_one({
            "email": email,
            "password_hash": hash_password(password),
            "full_name": args.name.strip(),
            "role": "officer",
            "created_at": datetime.now(UTC),
            "is_active": True,
        })
    except DuplicateKeyError:
        print("An account with this email already exists.", file=sys.stderr)
        return 1
    print(f"Officer account created for {email}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
