#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NestSAR single-file launcher for Kaggle and local virtual environments.

The complete verified trainer is stored in small repository source fragments so
this launcher remains easy to download and execute from a Kaggle cell. When the
repository was cloned, fragments are read locally. When only ``nestsar.py`` was
downloaded, the same fragments are fetched from the public main branch.

The reconstructed packed trainer verifies its checksum, then verifies and runs
the complete readable implementation. All command-line arguments are forwarded
unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

PACKED_SHA256 = "ad211007ae7ce6f166ac57250231ace68556e92b23cf990d2abacda20767a1f9"
PART_NAMES = (
    "raw00.b64",
    "raw01a.b64",
    "raw01b.b64",
    "raw02a.b64",
    "raw02b.b64",
)
DEFAULT_REF = "main"
DEFAULT_RAW_BASE = "https://raw.githubusercontent.com/rombaldivia/NestSAR"


def _local_parts_dir() -> Path:
    return Path(__file__).resolve().parent / ".assembly"


def _read_local_parts() -> list[str] | None:
    root = _local_parts_dir()
    paths = [root / name for name in PART_NAMES]
    if not all(path.is_file() for path in paths):
        return None
    return [path.read_text(encoding="ascii").strip() for path in paths]


def _download_parts() -> list[str]:
    ref = os.environ.get("NESTSAR_SOURCE_REF", DEFAULT_REF).strip() or DEFAULT_REF
    base = os.environ.get("NESTSAR_RAW_BASE", DEFAULT_RAW_BASE).rstrip("/")
    parts: list[str] = []
    for name in PART_NAMES:
        url = f"{base}/{ref}/.assembly/{name}"
        request = Request(url, headers={"User-Agent": "NestSAR-launcher/1.0"})
        try:
            with urlopen(request, timeout=60) as response:
                parts.append(response.read().decode("ascii").strip())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                "No se pudo descargar el entrenador NestSAR. En Kaggle activa "
                "Internet para esta primera descarga, o clona el repositorio "
                "completo antes de desactivar Internet. URL fallida: " + url
            ) from exc
    return parts


def _load_packed_trainer() -> bytes:
    parts = _read_local_parts()
    source = "local repository"
    if parts is None:
        parts = _download_parts()
        source = "GitHub main"

    try:
        packed = base64.b64decode("".join(parts), validate=True)
    except Exception as exc:
        raise RuntimeError("Los fragmentos del entrenador no forman Base64 válido.") from exc

    observed = hashlib.sha256(packed).hexdigest()
    if observed != PACKED_SHA256:
        raise RuntimeError(
            "Checksum incorrecto al reconstruir nestsar.py: "
            f"esperado={PACKED_SHA256}, obtenido={observed}"
        )

    print(f"NestSAR trainer verified from {source}: {observed[:12]}", flush=True)
    return packed


def main() -> None:
    packed = _load_packed_trainer()
    exec(compile(packed, __file__, "exec"), globals(), globals())


if __name__ == "__main__":
    main()
