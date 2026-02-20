"""Python helper utilities for the Telum ONNX Runtime plugin EP package.

This follows the ONNX Runtime plugin EP packaging guidance:
- Provide helper functions that tell consumers where the shared library lives.
- Do not bundle ONNX Runtime itself in this package.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List


_EP_NAME = "TelumPluginExecutionProvider"


def get_ep_name() -> str:
    """Return the canonical EP name to use with ORT APIs."""

    return _EP_NAME


def get_ep_names() -> List[str]:
    """Return the list of EP names exposed by this plugin package."""

    return [_EP_NAME]


def _platform_library_filenames(base: str) -> List[str]:
    # NOTE: CMake builds as:
    # - Linux:   lib<name>.so
    # - macOS:   lib<name>.dylib
    # - Windows: <name>.dll
    if sys.platform.startswith("win"):
        return [f"{base}.dll"]
    if sys.platform == "darwin":
        return [f"lib{base}.dylib"]
    return [f"lib{base}.so"]


def get_library_path() -> str:
    """Return an absolute filesystem path to the plugin EP shared library.

    Expected package layout:
      onnxruntime_ep_telum/
        __init__.py
        lib/
          libtelum_plugin_ep.so (Linux)
          libtelum_plugin_ep.dylib (macOS)
          telum_plugin_ep.dll (Windows)
    """

    pkg_dir = Path(__file__).resolve().parent
    lib_dir = pkg_dir / "lib"

    base = "telum_plugin_ep"
    candidates = [lib_dir / name for name in _platform_library_filenames(base)]

    for p in candidates:
        if p.is_file():
            return str(p)

    # Fallback: pick any file in lib/ with a plausible shared-library suffix.
    if lib_dir.is_dir():
        suffixes = (".so", ".dylib", ".dll")
        for p in sorted(lib_dir.iterdir()):
            if p.is_file() and p.name.endswith(suffixes):
                return str(p)

    raise FileNotFoundError(
        f"Telum plugin EP shared library not found. Looked in: {lib_dir}. "
        "Expected the package to include the built library under onnxruntime_ep_telum/lib/."
    )


def _debug_dump_paths() -> str:
    # Useful for quick diagnostics in bug reports.
    pkg_dir = Path(__file__).resolve().parent
    lib_dir = pkg_dir / "lib"
    entries = []
    for p in [pkg_dir, lib_dir]:
        if not p.exists():
            entries.append(f"{p} (missing)")
            continue
        if p.is_file():
            entries.append(f"{p} (file)")
            continue
        entries.append(f"{p}/")
        for c in sorted(p.iterdir()):
            entries.append(f"  - {c.name}")
    return os.linesep.join(entries)
