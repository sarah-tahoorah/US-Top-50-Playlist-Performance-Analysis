"""Authentication gate for the Streamlit dashboard.

Prefers `streamlit-authenticator`; falls back to an equivalent bcrypt-checked
session_state gate if the library is not installed. Credentials live in
config/credentials.yaml as bcrypt hashes — never plaintext.
"""
from __future__ import annotations

import os

import bcrypt
import streamlit as st
import yaml

CONFIG_PATH = os.path.join("config", "credentials.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _verify(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except ValueError:
        return False


def _fallback_login(config: dict) -> tuple[bool, str]:
    """Minimal session_state gate used when streamlit-authenticator is absent."""
    users = config["credentials"]["usernames"]
    if st.session_state.get("auth_ok"):
        return True, st.session_state.get("auth_name", "")

    st.markdown("### Sign in")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        record = users.get(username.strip())
        if record and _verify(password, record["password"]):
            st.session_state["auth_ok"] = True
            st.session_state["auth_name"] = record.get("name", username)
            st.session_state["auth_user"] = username
            st.rerun()
        else:
            st.error("Incorrect username or password.")
    return False, ""


def login(config_path: str = CONFIG_PATH) -> tuple[bool, str, object | None]:
    """Render the login screen. Returns (authenticated, display_name, authenticator)."""
    config = load_config(config_path)
    try:
        import streamlit_authenticator as stauth
    except Exception:
        ok, name = _fallback_login(config)
        return ok, name, None

    authenticator = stauth.Authenticate(
        config["credentials"],
        config["cookie"]["name"],
        config["cookie"]["key"],
        config["cookie"]["expiry_days"],
    )
    try:
        authenticator.login(location="main")
    except Exception as exc:  # pragma: no cover - surfaced to the user
        st.error(f"Authentication error: {exc}")

    status = st.session_state.get("authentication_status")
    name = st.session_state.get("name") or ""
    if status is False:
        st.error("Incorrect username or password.")
    elif status is None:
        st.info("Please enter your credentials to access the dashboard.")
    return bool(status), name, authenticator


def logout(authenticator: object | None) -> None:
    if authenticator is not None:
        authenticator.logout("Logout", "sidebar")
    elif st.sidebar.button("Logout"):
        for key in ["auth_ok", "auth_name", "auth_user"]:
            st.session_state.pop(key, None)
        st.rerun()
