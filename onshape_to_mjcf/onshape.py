"""Minimal HMAC-signed Onshape REST client."""

from __future__ import annotations

import base64
import datetime
import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path

import requests
from dotenv import load_dotenv

# Credentials live in the repo-root .env (see .env.example).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Onshape:
    """Minimal HMAC-signed Onshape REST client."""

    def __init__(self) -> None:
        self.api = os.environ["ONSHAPE_API"].rstrip("/")
        self.key = os.environ["ONSHAPE_ACCESS_KEY"]
        self.secret = os.environ["ONSHAPE_SECRET_KEY"]

    def request(self, method: str, path: str, query: str = "",
                body: dict | None = None, raw: bool = False):
        date = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%a, %d %b %Y %H:%M:%S GMT")
        nonce = "".join(
            secrets.choice("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
            for _ in range(25)
        )
        to_sign = f"{method}\n{nonce}\n{date}\napplication/json\n{path}\n{query}\n".lower().encode()
        signature = base64.b64encode(
            hmac.new(self.secret.encode(), to_sign, hashlib.sha256).digest()
        ).decode()
        headers = {
            "Date": date,
            "On-Nonce": nonce,
            "Authorization": f"On {self.key}:HmacSHA256:{signature}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "onshape-to-mjcf",
        }
        data = json.dumps(body) if body is not None else None
        response = requests.request(
            method,
            self.api + path + (("?" + query) if query else ""),
            headers=headers,
            data=data,
            timeout=120,
        )
        if raw:
            return response
        try:
            return response.status_code, response.json()
        except Exception:
            return response.status_code, response.text

    def get(self, path: str, query: str = ""):
        return self.request("GET", path, query)

    def post(self, path: str, body: dict, query: str = ""):
        return self.request("POST", path, query, body)

    def delete(self, path: str):
        return self.request("DELETE", path)
