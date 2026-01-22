#!/usr/bin/env python3
import os
import re
import subprocess
import sys
from pathlib import Path


WHEEL_PYTAG_RE = re.compile(r"(?:^|-)cp(?P<major>\d)(?P<minor>\d{1,2})(?:-|\.whl$)")


def run(cmd: list[str], *, env: dict[str, str] | None = None) -> None:
    print(f"+ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, env=env)


def wheel_python_tag(wheel_name: str) -> tuple[int, int]:
    m = WHEEL_PYTAG_RE.search(wheel_name)
    if not m:
        raise ValueError(
            f"Could not infer Python tag from wheel filename: {wheel_name}\n"
            "Expected wheel filename to contain e.g. '-cp310-' or '-cp311-'."
        )
    major = int(m.group("major"))
    minor = int(m.group("minor"))
    return major, minor


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    dist_dir = repo_root / "dist"
    if not dist_dir.is_dir():
        print(f"ERROR: dist/ not found at {dist_dir}", file=sys.stderr)
        return 1

    wheels = sorted(dist_dir.glob("*.whl"))
    if not wheels:
        print("ERROR: no wheels found in dist/", file=sys.stderr)
        return 1

    overall_ok = True
    for wheel in wheels:
        major, minor = wheel_python_tag(wheel.name)
        pytag = f"cp{major}{minor}"

        print("\n" + "=" * 80)
        print(f"Wheel: {wheel.name}")
        print(f"Needs: {pytag}")

        # Activate the venv
        venv_dir = repo_root / ".venv"  # or better: per-wheel unique dir on Windows
        run(["uv", "venv", "--python", f"{major}.{minor}", "--clear", str(venv_dir)])
        env = os.environ.copy()
        env["VIRTUAL_ENV"] = str(venv_dir)

        # Install deps
        run(["uv", "pip", "install", "--group", "dev"], env=env)
        run(
            [
                "uv",
                "pip",
                "install",
                "--find-links",
                str(dist_dir),
                "--force-reinstall",
                str(wheel),
            ],
            env=env,
        )

        run(["uv", "run", "--active", "--no-sync", "pytest", "-q"], env=env)

    print("\n" + "=" * 80)
    if overall_ok:
        print("All wheel tests passed.")
        return 0
    print("Some wheels were skipped (missing interpreters) or failed.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
