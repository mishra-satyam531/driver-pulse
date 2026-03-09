from __future__ import annotations
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "driver_pulse_submission.zip"

INCLUDE = [
    "app/",
    "backend/",
    "data/processed_outputs/",
    "requirements.txt",
    "README.md",
    "design_doc.md",
    "progress_log.md",
]

EXCLUDE_PREFIXES = [
    ".venv/",
    ".git_disabled/",
]

def should_include(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    for p in EXCLUDE_PREFIXES:
        if rel.startswith(p):
            return False
    return True

def package() -> Path:
    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for item in INCLUDE:
            p = ROOT / item
            if p.is_file():
                if should_include(p):
                    zf.write(p, p.relative_to(ROOT).as_posix())
            elif p.is_dir():
                for sub in p.rglob("*"):
                    if sub.is_file() and should_include(sub):
                        zf.write(sub, sub.relative_to(ROOT).as_posix())
    print(f"Submission package -> {OUT}")
    return OUT

if __name__ == "__main__":
    package()
