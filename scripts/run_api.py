from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="Chạy API trợ lý pháp luật")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("legal_agent.api.main:app", host=args.host, port=args.port,
                reload=args.reload)


if __name__ == "__main__":
    main()
