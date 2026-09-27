"""CLI entry point to launch the Concealed API server."""

from __future__ import annotations

import argparse
import os
import sys
import uvicorn


def main() -> None:
    default_port = int(os.environ.get("PORT", 8001))
    parser = argparse.ArgumentParser(description="Concealed Image Obfuscation API Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Network host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=default_port, help=f"Server port (default: {default_port})")
    parser.add_argument("--reload", action="store_true", help="Enable automatic code reload for development")
    parser.add_argument("--workers", type=int, default=1, help="Number of uvicorn worker processes")
    args = parser.parse_args()

    print(
        f"+===========================================================+\n"
        f"|  Starting Concealed Obfuscation Server on {args.host}:{args.port}  |\n"
        f"|  Interactive API Docs: http://{args.host}:{args.port}/docs        |\n"
        f"+===========================================================+",
        flush=True,
    )

    uvicorn.run(
        "concealed.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()
