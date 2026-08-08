"""Create or update config/credentials.yaml with bcrypt-hashed passwords.

Usage:
    python -m config.make_credentials                 # writes the demo account
    python -m config.make_credentials alice S3cret!   # adds/updates a user
"""
from __future__ import annotations

import os
import sys

import bcrypt
import yaml

CONFIG_PATH = os.path.join("config", "credentials.yaml")
DEMO = [("admin", "Atlantic2024!", "Atlantic Analyst", "analyst@atlanticrecords.example")]


def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def main() -> None:
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r", encoding="utf-8") as fh:
            config = yaml.safe_load(fh) or {}
    else:
        config = {}

    config.setdefault("credentials", {}).setdefault("usernames", {})
    config.setdefault("cookie", {
        "name": "playlist_analytics_auth",
        "key": "change-this-cookie-signing-key",
        "expiry_days": 1,
    })

    if len(sys.argv) >= 3:
        username, password = sys.argv[1], sys.argv[2]
        name = sys.argv[3] if len(sys.argv) > 3 else username.title()
        entries = [(username, password, name, f"{username}@example.com")]
    else:
        entries = DEMO

    for username, password, name, email in entries:
        config["credentials"]["usernames"][username] = {
            "name": name, "email": email, "password": hash_password(password),
        }
        print(f"Set credentials for {username!r}")

    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False)
    print(f"Wrote {CONFIG_PATH}")


if __name__ == "__main__":
    main()
