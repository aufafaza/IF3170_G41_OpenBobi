import argparse
import os
import sys
import webbrowser
from pathlib import Path

from .dashboard import build_dashboard


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    parser = argparse.ArgumentParser(
        prog="python -m src.visualization",
        description="Buat dashboard HTML dari hasil eksperimen local search.",
    )
    parser.add_argument("-i", "--input", default="output", help="folder output yang berisi experiments/ dan solutions/")
    parser.add_argument("-o", "--out", default=None, help="lokasi file HTML (default: <input>/dashboard.html)")
    parser.add_argument("--open", action="store_true", help="buka dashboard di browser setelah dibuat")
    args = parser.parse_args(argv)

    try:
        path, warnings = build_dashboard(args.input, args.out)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    for message in warnings:
        print(f"[WARN] {message}", file=sys.stderr)
    print(f"[OK] Dashboard dibuat: {os.path.abspath(path)}")
    if args.open:
        webbrowser.open(Path(path).resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())