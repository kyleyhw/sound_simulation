"""Entry point for the FastAPI/Socket.IO acoustic simulation server.

Run from the project root:
    python scripts/run_ui_server.py

The frontend (Vite dev server, port 3000) proxies ``/socket.io`` to this
server on port 8001.
"""

from __future__ import annotations

import os
import sys

import uvicorn


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.abspath(os.path.join(here, "..", "src"))

    # Ensure the package is importable when running this script directly.
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    host = os.environ.get("ACOUSTIC_HOST", "127.0.0.1")
    port = int(os.environ.get("ACOUSTIC_PORT", "8001"))

    print(f"[acoustic] Starting backend at http://{host}:{port}")
    print("[acoustic] Frontend dev server: http://127.0.0.1:3000 (vite)")
    uvicorn.run(
        "acoustic_system.app.main:socket_app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
