"""Export the Gradus dog CAD from Onshape (STEP, STL, PNG).

The Onshape document is built by ``cad/gradus_dog.fs`` (also in the document's
Feature Studio).  This script asks Onshape to translate the Part Studio and
downloads the result.

Credentials come from the environment::

    export ONSHAPE_API=https://cad.onshape.com
    export ONSHAPE_ACCESS_KEY=...
    export ONSHAPE_SECRET_KEY=...

Usage::

    python -m cad.export_onshape                       # STEP, STL and two PNGs
    python -m cad.export_onshape --formats step stl    # only those
    python -m cad.export_onshape --out /tmp/dog        # somewhere else
    python -m cad.export_onshape --attempts 20         # keep retrying (Onshape outages)

Onshape sometimes leaves a translation in ACTIVE for a long time or answers
500 while it is busy, so every format is retried.  Check
https://status.onshape.com/ before blaming the document.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import hmac
import json
import os
import secrets
import time
from pathlib import Path

import requests

DOCUMENT_ID = "572c4a586468c3b5310ca798"
WORKSPACE_ID = "3b9ccafe4a26826d5aea33e4"
PART_STUDIO_ID = "9adb8759bcda810b416055f0"

# (format name, file suffix, extra translation options, view name)
FORMATS = {
    "step": ("STEP", ".step", {"stepVersion": "AP242"}, None),
    "stl": ("STL", ".stl", {"meshParams": {"angularTolerance": 0.001, "distanceTolerance": 0.001,
                                          "maximumChordLength": 0.01, "resolution": "FINE",
                                          "unit": "MILLIMETER"}}, None),
    "isometric": ("PNG", ".png", {"pngOptions": {"backgroundColor": [255, 255, 255, 255],
                                                 "colorOrientation": "Z_AXIS", "showMassProperties": False,
                                                 "showUnits": False, "viewAxis": "Z_AXIS",
                                                 "viewRotation": 45.0}}, "gradus_dog_isometric.png"),
    "front": ("PNG", ".png", {"pngOptions": {"backgroundColor": [255, 255, 255, 255],
                                             "colorOrientation": "Z_AXIS", "showMassProperties": False,
                                             "showUnits": False, "viewAxis": "Y_AXIS",
                                             "viewRotation": 0.0}}, "gradus_dog_front.png"),
}


class Onshape:
    """Minimal HMAC-signed Onshape REST client."""

    def __init__(self) -> None:
        self.api = os.environ["ONSHAPE_API"].rstrip("/")
        self.key = os.environ["ONSHAPE_ACCESS_KEY"]
        self.secret = os.environ["ONSHAPE_SECRET_KEY"]

    def request(self, method: str, path: str, query: str = "", body: dict | None = None, raw: bool = False):
        date = datetime.datetime.utcnow().strftime("%a, %d %b %Y %H:%M:%S GMT")
        nonce = "".join(secrets.choice("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
                         for _ in range(25))
        to_sign = f"{method}\n{nonce}\n{date}\napplication/json\n{path}\n{query}\n".lower().encode()
        signature = base64.b64encode(hmac.new(self.secret.encode(), to_sign, hashlib.sha256).digest()).decode()
        headers = {
            "Date": date,
            "On-Nonce": nonce,
            "Authorization": f"On {self.key}:HmacSHA256:{signature}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "gradus-simulation-export",
        }
        data = json.dumps(body) if body is not None else None
        response = requests.request(method, self.api + path + (("?" + query) if query else ""),
                                    headers=headers, data=data, timeout=120)
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


def start_translation(client: Onshape, fmt: str, extra: dict) -> str | None:
    body = {"formatName": fmt, "storeInDocument": False, "allowFaultyParts": True}
    body.update(extra)
    path = f"/api/v16/partstudios/d/{DOCUMENT_ID}/w/{WORKSPACE_ID}/e/{PART_STUDIO_ID}/translations"
    status, payload = client.post(path, body)
    if status != 200:
        print(f"  {fmt}: translation request failed ({status}) {str(payload)[:160]}")
        return None
    return payload.get("id")


def wait_for_translation(client: Onshape, tid: str, timeout: float = 300.0) -> dict | None:
    deadline = time.time() + timeout
    delay = 3.0
    state = None
    while time.time() < deadline:
        status, payload = client.get(f"/api/v16/translations/{tid}")
        if status != 200:
            time.sleep(delay)
            continue
        if payload.get("requestState") != state:
            state = payload.get("requestState")
            print(f"  state: {state}")
        if state == "DONE":
            return payload
        if state == "FAILED":
            print(f"  translation failed: {payload.get('failureReason')}")
            return None
        time.sleep(delay)
        delay = min(delay * 1.4, 15.0)
    print("  translation still active, giving up on this attempt")
    return None


def download(client: Onshape, data_id: str, path: Path) -> bool:
    response = client.request("GET", f"/api/v16/documents/d/{DOCUMENT_ID}/externaldata/{data_id}", raw=True)
    if response.status_code != 200:
        print(f"  download failed ({response.status_code})")
        return False
    path.write_bytes(response.content)
    print(f"  wrote {path} ({path.stat().st_size:,} bytes)")
    return True


def export(client: Onshape, names: list[str], out: Path, attempts: int) -> bool:
    out.mkdir(parents=True, exist_ok=True)
    pending = list(names)
    for attempt in range(1, attempts + 1):
        if not pending:
            return True
        print(f"attempt {attempt}: {', '.join(pending)}")
        for name in list(pending):
            fmt, suffix, extra, filename = FORMATS[name]
            tid = start_translation(client, fmt, extra)
            if tid is None:
                continue
            done = wait_for_translation(client, tid)
            if done is None:
                continue
            ids = done.get("resultExternalDataIds") or []
            if not ids:
                print("  no external data id in the result")
                continue
            target = out / (filename or f"gradus_dog{suffix}")
            if download(client, ids[0], target):
                pending.remove(name)
        if pending:
            time.sleep(20)
    return not pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("/home/daniel/Downloads/gradus_dog"))
    parser.add_argument("--formats", nargs="+", choices=sorted(FORMATS), default=sorted(FORMATS))
    parser.add_argument("--attempts", type=int, default=6, help="how often to retry each format")
    args = parser.parse_args()

    client = Onshape()
    ok = export(client, list(args.formats), args.out, args.attempts)
    if not ok:
        raise SystemExit("some exports did not finish; check https://status.onshape.com/ and retry")
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
