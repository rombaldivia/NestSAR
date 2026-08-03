#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NestSAR single-file launcher for Kaggle and local virtual environments.

The complete readable trainer is stored as verified compressed source fragments
inside this repository. When the repository is cloned, the launcher reads those
fragments locally and can run without Internet. When only ``nestsar.py`` is
downloaded, the launcher fetches the same fragments from the public main branch.

Every launch verifies both the compressed payload and the reconstructed Python
source before executing it. All command-line arguments are forwarded unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import lzma
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

SOURCE_SHA256 = "8aa931b9423bbe4aaba2258563021797f738619ae6fd5f9a227ca9239dfb49d4"
COMPRESSED_SHA256 = "14278657c69fe98278134d4456c3e7c26b835658927d83e6f408170aca27bc01"
PART_NAMES = tuple(f"chunk{index:03d}.b64" for index in range(12))
DEFAULT_REF = "main"
DEFAULT_RAW_BASE = "https://raw.githubusercontent.com/rombaldivia/NestSAR"


def _local_source_dir() -> Path:
    return Path(__file__).resolve().parent / ".source"


def _read_local_parts() -> list[str] | None:
    root = _local_source_dir()
    paths = [root / name for name in PART_NAMES]
    if not all(path.is_file() for path in paths):
        return None
    return [path.read_text(encoding="ascii").strip() for path in paths]


def _download_parts() -> list[str]:
    ref = os.environ.get("NESTSAR_SOURCE_REF", DEFAULT_REF).strip() or DEFAULT_REF
    base = os.environ.get("NESTSAR_RAW_BASE", DEFAULT_RAW_BASE).rstrip("/")
    parts: list[str] = []

    for name in PART_NAMES:
        url = f"{base}/{ref}/.source/{name}"
        request = Request(url, headers={"User-Agent": "NestSAR-launcher/1.0"})
        try:
            with urlopen(request, timeout=60) as response:
                parts.append(response.read().decode("ascii").strip())
        except (HTTPError, URLError, TimeoutError) as exc:
            raise RuntimeError(
                "No se pudo descargar el código fuente completo de NestSAR. "
                "En Kaggle activa Internet para esta primera descarga, o clona "
                "el repositorio completo y luego ejecútalo sin Internet. "
                f"URL fallida: {url}"
            ) from exc

    return parts


def _reconstruct_source() -> tuple[bytes, str]:
    parts = _read_local_parts()
    origin = "repositorio local"
    if parts is None:
        parts = _download_parts()
        origin = "GitHub main"

    try:
        compressed = base64.b64decode("".join(parts), validate=True)
    except Exception as exc:
        raise RuntimeError("Los fragmentos de NestSAR no forman Base64 válido.") from exc

    compressed_hash = hashlib.sha256(compressed).hexdigest()
    if compressed_hash != COMPRESSED_SHA256:
        raise RuntimeError(
            "Checksum incorrecto del paquete NestSAR: "
            f"esperado={COMPRESSED_SHA256}, obtenido={compressed_hash}"
        )

    try:
        source = lzma.decompress(compressed)
    except lzma.LZMAError as exc:
        raise RuntimeError("No se pudo descomprimir el código fuente de NestSAR.") from exc

    source_hash = hashlib.sha256(source).hexdigest()
    if source_hash != SOURCE_SHA256:
        raise RuntimeError(
            "Checksum incorrecto del código NestSAR reconstruido: "
            f"esperado={SOURCE_SHA256}, obtenido={source_hash}"
        )

    return source, origin


def _export_source(source: bytes) -> bool:
    if "--export-source" not in sys.argv:
        return False

    position = sys.argv.index("--export-source")
    try:
        destination = Path(sys.argv[position + 1]).expanduser().resolve()
    except IndexError as exc:
        raise SystemExit("--export-source requiere una ruta de destino") from exc

    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(source)
    print(destination)
    return True


def main() -> None:
    source, origin = _reconstruct_source()
    print(
        f"NestSAR verificado desde {origin}: {SOURCE_SHA256[:12]}",
        flush=True,
    )

    if _export_source(source):
        return

    exec(compile(source, __file__, "exec"), globals(), globals())


if __name__ == "__main__":
    main()
