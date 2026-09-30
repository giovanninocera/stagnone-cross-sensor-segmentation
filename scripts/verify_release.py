#!/usr/bin/env python3
"""Verify release integrity and enforce the public-data disclosure boundary."""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECKSUM_FILE = ROOT / "SHA256SUMS.txt"

FORBIDDEN_EXTENSIONS = {
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".jp2", ".bmp", ".gif",
    ".webp", ".vrt", ".img", ".h5", ".hdf5", ".pt", ".pth", ".ckpt",
    ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".geojson", ".gpkg",
    ".shp", ".dbf", ".shx", ".qgz", ".qgs", ".docx", ".pdf", ".pptx",
    ".zip", ".7z", ".rar", ".sqlite", ".db",
}
ALLOWED_EXTENSIONS = {"", ".md", ".txt", ".csv", ".json", ".py", ".cff", ".yml", ".yaml"}
MAGIC_SIGNATURES = {
    b"\x89PNG\r\n\x1a\n": "PNG",
    b"\xff\xd8\xff": "JPEG",
    b"II*\x00": "TIFF little-endian",
    b"MM\x00*": "TIFF big-endian",
    b"PK\x03\x04": "ZIP/Office/OpenXML",
    b"\x89HDF\r\n\x1a\n": "HDF5",
}
# A Windows drive-root path is private; the ``s://`` portion of an HTTPS URL is not.
LOCAL_PATH_PATTERN = re.compile(
    r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]|/Users/|/home/|OneDrive)",
    re.I,
)
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}"),
    "OpenAI key": re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    "AWS access key": re.compile(r"AKIA[0-9A-Z]{16}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_expected() -> dict[str, str]:
    expected: dict[str, str] = {}
    pattern = re.compile(r"^([0-9a-f]{64})  (.+)$")
    for line_number, raw in enumerate(CHECKSUM_FILE.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not raw.strip():
            continue
        match = pattern.match(raw)
        if not match:
            raise ValueError(f"Invalid SHA256SUMS line {line_number}: {raw!r}")
        relative = match.group(2).replace("\\", "/")
        if relative in expected:
            raise ValueError(f"Duplicate checksum entry: {relative}")
        expected[relative] = match.group(1)
    return expected


def main() -> int:
    failures: list[str] = []
    if not CHECKSUM_FILE.is_file():
        print("FAIL: SHA256SUMS.txt is missing", file=sys.stderr)
        return 1
    try:
        expected = load_expected()
    except ValueError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1

    actual_files = {
        path.relative_to(ROOT).as_posix(): path
        for path in ROOT.rglob("*")
        if path.is_file() and path != CHECKSUM_FILE and ".git" not in path.parts
    }
    failures.extend(f"missing listed file: {item}" for item in sorted(set(expected) - set(actual_files)))
    failures.extend(f"unlisted extra file: {item}" for item in sorted(set(actual_files) - set(expected)))

    for relative, path in sorted(actual_files.items()):
        suffix = path.suffix.lower()
        if suffix in FORBIDDEN_EXTENSIONS:
            failures.append(f"forbidden extension {suffix}: {relative}")
        if suffix not in ALLOWED_EXTENSIONS:
            failures.append(f"extension not allowlisted {suffix}: {relative}")
        if path.stat().st_size > 10 * 1024 * 1024:
            failures.append(f"unexpected file larger than 10 MiB: {relative}")

        raw = path.read_bytes()
        prefix = raw[:16]
        for signature, label in MAGIC_SIGNATURES.items():
            if prefix.startswith(signature):
                failures.append(f"forbidden {label} magic bytes: {relative}")
        if b"\x00" in prefix:
            failures.append(f"binary NUL byte detected: {relative}")

        try:
            text = raw.decode("utf-8-sig")
        except UnicodeDecodeError:
            failures.append(f"non-UTF-8 text payload: {relative}")
            text = ""
        if relative != "scripts/verify_release.py" and LOCAL_PATH_PATTERN.search(text):
            failures.append(f"private absolute path detected: {relative}")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                failures.append(f"possible {label} detected: {relative}")

        wanted = expected.get(relative)
        if wanted is not None and sha256(path) != wanted:
            failures.append(f"SHA-256 mismatch: {relative}")

    if failures:
        for failure in failures:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1

    summary_check = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "reproduce_summary.py"), "--check"],
        cwd=ROOT,
        check=False,
    )
    if summary_check.returncode != 0:
        print("FAIL: numerical-summary check failed", file=sys.stderr)
        return summary_check.returncode

    total_bytes = sum(path.stat().st_size for path in actual_files.values())
    print(
        f"PASS: {len(actual_files)} release files verified; {total_bytes} bytes; "
        "no forbidden binary/geospatial content, private paths, or common credential signatures detected"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
