from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Chạy web UI trợ lý pháp luật")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--host", default="localhost")
    args = parser.parse_args()

    command = [
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app" / "streamlit_app.py"),
        "--server.port", str(args.port),
        "--server.address", args.host,
        "--browser.gatherUsageStats", "false",
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
