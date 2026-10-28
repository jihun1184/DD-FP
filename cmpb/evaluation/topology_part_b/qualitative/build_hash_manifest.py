#!/usr/bin/env python3
"""Rebuild SHA256SUMS.txt for a completed qualitative reserve tree."""

from __future__ import annotations

import argparse
from pathlib import Path

from qualitative_common import write_sha256_manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    if not args.root.is_dir():
        raise NotADirectoryError(args.root)
    path = write_sha256_manifest(args.root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
