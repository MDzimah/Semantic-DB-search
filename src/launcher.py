"""Launcher for running the Streamlit app from a normal Python process.

This is the entry point that will later be wrapped into a Windows executable.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys


def resource_path(relative_path: str) -> Path:
	base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
	return base_path / relative_path


def main() -> int:
	app_path = resource_path("app.py")
	command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
	process = subprocess.run(command, check=False)
	return int(process.returncode)


if __name__ == "__main__":
	raise SystemExit(main())
