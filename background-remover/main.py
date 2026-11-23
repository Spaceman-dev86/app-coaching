#!/usr/bin/env python3
"""Entry point: HTML UI on http://127.0.0.1:7860 (FastAPI).

Optional Gradio UI: python -c "from gradio_app import main; main()"
"""

from __future__ import annotations


def main() -> None:
    from server import main as serve

    serve()


if __name__ == "__main__":
    main()
