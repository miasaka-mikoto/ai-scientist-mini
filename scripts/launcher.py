"""PyInstaller entry point for AI Scientist Mini.

The package's ``__main__`` module uses relative imports, so PyInstaller must
start through this tiny top-level shim rather than executing that file as an
orphan script.
"""

from ai_scientist_mini.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
