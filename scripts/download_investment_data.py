#!/usr/bin/env python
"""Download and verify the latest chenditc/investment_data Qlib archive.

Direct github.com release URLs are not reachable from this host; the GitHub
API release-asset endpoint is reachable, so this script uses that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import time
from pathlib import Path

import requests

API = "https://api.github.com/repos/chenditc/investment_data"
TARGET = Path("D:/qlib_data/cn_data_investment_data")


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"})
    return session


def _release(session: requests.Session, tag: str | None) -> dict:
    url = f"{API}/releases/latest" if not tag else f"{API}/releases/tags/{tag}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _download_asset(session: requests.Session, asset: dict, target: Path) -> None:
    url = f"{API}/releases/assets/{asset['id']}"
    headers = {"Accept": "application/octet-stream"}
    expected_size = int(asset["size"])
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_suffix(target.suffix + ".part")
    existing = part.stat().st_size if part.exists() else 0
    if existing == expected_size:
        part.replace(target)
        return
    if existing > expected_size:
        part.unlink()
        existing = 0

    for attempt in range(4):
        try:
            request_headers = {**headers}
            if existing:
                request_headers["Range"] = f"bytes={existing}-"
            with session.get(url, headers=request_headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                if response.status_code not in (200, 206):
                    response.raise_for_status()
                mode = "ab" if existing and response.status_code == 206 else "wb"
                if mode == "wb":
                    existing = 0
                written = existing
                last_report = time.time()
                with part.open(mode) as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        written += len(chunk)
                        if time.time() - last_report > 10:
                            print(
                                f"[download] {written / 1e9:.2f} / {expected_size / 1e9:.2f} GB",
                                flush=True,
                            )
                            last_report = time.time()
            if part.stat().st_size == expected_size:
                part.replace(target)
                return
            existing = part.stat().st_size
        except Exception as exc:
            print(f"[download] attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(3.0 * (attempt + 1))
    raise RuntimeError(f"failed to download {target}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default=None)
    parser.add_argument("--target", default=str(TARGET))
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)

    target = Path(args.target)
    session = _session()
    release = _release(session, args.tag)
    tag = release["tag_name"]
    assets = {asset["name"]: asset for asset in release["assets"]}
    print(f"[release] {tag} assets={list(assets)}", flush=True)
    manifest_path = target / "qlib_bin.manifest.json"
    _download_asset(session, assets["qlib_bin.manifest.json"], manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)

    archive = target / "qlib_bin.tar.gz"
    _download_asset(session, assets["qlib_bin.tar.gz"], archive)
    actual_size = archive.stat().st_size
    actual_hash = _sha256(archive)
    expected_hash = manifest["archive_sha256"].split(":", 1)[-1]
    print(f"[verify] size={actual_size} sha256={actual_hash}", flush=True)
    if actual_size != manifest["archive_size_bytes"]:
        raise SystemExit("size mismatch")
    if actual_hash != expected_hash:
        raise SystemExit("sha256 mismatch")
    print("[verify] archive OK", flush=True)

    if args.extract:
        extract_dir = target / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        destination = extract_dir.resolve()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            print(f"[extract] {len(members)} members", flush=True)
            for member in members:
                resolved = (destination / member.name).resolve()
                if destination not in resolved.parents and resolved != destination:
                    raise SystemExit(f"unsafe archive member: {member.name}")
                if member.issym() or member.islnk():
                    raise SystemExit(f"unsafe archive link: {member.name}")
            for member in members[:20]:
                print(f"[extract] {member.name}", flush=True)
            tar.extractall(extract_dir, members=members)
        print(f"[extract] -> {extract_dir}", flush=True)


if __name__ == "__main__":
    main()
