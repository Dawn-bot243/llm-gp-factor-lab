#!/usr/bin/env python
"""Parallel-range downloader for the investment_data Qlib archive.

Keeps an existing sequential ``.part`` prefix, downloads the remaining ranges
with several connections, then verifies the manifest SHA256 before extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

API = "https://api.github.com/repos/chenditc/investment_data"
TARGET = Path("D:/qlib_data/cn_data_investment_data")
_lock = threading.Lock()
_done = 0


def _session() -> requests.Session:
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    return session


def _release(session: requests.Session, tag: str | None) -> dict:
    url = f"{API}/releases/latest" if not tag else f"{API}/releases/tags/{tag}"
    response = session.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


def _asset(session: requests.Session, asset: dict, target: Path) -> None:
    url = f"{API}/releases/assets/{asset['id']}"
    if target.exists() and target.stat().st_size == asset["size"]:
        return
    response = session.get(
        url,
        headers={"Accept": "application/octet-stream"},
        timeout=(30, 300),
        stream=True,
        allow_redirects=True,
    )
    response.raise_for_status()
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)


def _range_worker(asset_id: int, start: int, end: int, path: Path, retries: int = 4) -> Path:
    expected = end - start + 1
    for attempt in range(retries):
        session = _session()
        try:
            url = f"{API}/releases/assets/{asset_id}"
            response = session.get(
                url,
                headers={
                    "Accept": "application/octet-stream",
                    "Range": f"bytes={start}-{end}",
                },
                timeout=(30, 300),
                stream=True,
                allow_redirects=True,
            )
            if response.status_code != 206:
                raise RuntimeError(f"unexpected status {response.status_code}")
            with path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            if path.stat().st_size != expected:
                raise RuntimeError(f"size {path.stat().st_size} != {expected}")
            return path
        except Exception as exc:
            print(f"[range] {start}-{end} attempt {attempt + 1} failed: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"range failed: {start}-{end}")


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
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--chunk-mb", type=int, default=32)
    parser.add_argument("--extract", action="store_true")
    args = parser.parse_args()
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(key, None)

    target = Path(args.target)
    target.mkdir(parents=True, exist_ok=True)
    session = _session()
    release = _release(session, args.tag)
    assets = {asset["name"]: asset for asset in release["assets"]}
    print(f"[release] {release['tag_name']}", flush=True)
    manifest_path = target / "qlib_bin.manifest.json"
    _asset(session, assets["qlib_bin.manifest.json"], manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    total = int(manifest["archive_size_bytes"])
    expected_hash = manifest["archive_sha256"].split(":", 1)[-1]
    archive = target / "qlib_bin.tar.gz"
    part = target / "qlib_bin.tar.gz.part"
    prefix = part.stat().st_size if part.exists() else 0
    if prefix > total:
        part.unlink()
        prefix = 0
    print(f"[download] total={total / 1e6:.1f}MB existing_prefix={prefix / 1e6:.1f}MB", flush=True)

    chunk_bytes = args.chunk_mb * 1024 * 1024
    ranges = [(start, min(start + chunk_bytes - 1, total - 1)) for start in range(prefix, total, chunk_bytes)]
    part_files: list[Path] = []
    for index, (start, end) in enumerate(ranges):
        part_files.append(target / f"qlib_bin.tar.gz.chunk{index:03d}")
    if ranges:
        with ThreadPoolExecutor(max_workers=min(args.workers, len(ranges))) as pool:
            futures = {
                pool.submit(_range_worker, assets["qlib_bin.tar.gz"]["id"], start, end, path): (start, end)
                for (start, end), path in zip(ranges, part_files)
            }
            for future in as_completed(futures):
                start, end = futures[future]
                future.result()
                with _lock:
                    global _done
                    _done += end - start + 1
                    print(
                        f"[download] +{_done / 1e6:.1f}MB / {(total - prefix) / 1e6:.1f}MB",
                        flush=True,
                    )

    with archive.open("wb") as output:
        if prefix:
            with part.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
        for path in part_files:
            with path.open("rb") as handle:
                while True:
                    chunk = handle.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
    actual_size = archive.stat().st_size
    actual_hash = _sha256(archive)
    print(f"[verify] size={actual_size} sha256={actual_hash}", flush=True)
    if actual_size != total or actual_hash != expected_hash:
        raise SystemExit("archive verification failed")
    print("[verify] archive OK", flush=True)
    part.unlink(missing_ok=True)
    for path in part_files:
        path.unlink(missing_ok=True)

    if args.extract:
        extract_dir = target / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        destination = extract_dir.resolve()
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for member in members:
                resolved = (destination / member.name).resolve()
                if destination not in resolved.parents and resolved != destination:
                    raise SystemExit(f"unsafe archive member: {member.name}")
                if member.issym() or member.islnk():
                    raise SystemExit(f"unsafe archive link: {member.name}")
            print(f"[extract] {len(members)} members -> {extract_dir}", flush=True)
            tar.extractall(extract_dir, members=members)


if __name__ == "__main__":
    main()
