"""Download Kaggle Playground Series S6E11 dataset via kagglehub and stage into data/raw/."""

import shutil
from pathlib import Path

import kagglehub

from config import COMPETITION_SLUG, DATA_RAW_DIR, ensure_dirs


def main() -> None:
    ensure_dirs()
    print(f"Downloading competition data: {COMPETITION_SLUG} ...")
    download_path = Path(kagglehub.competition_download(COMPETITION_SLUG))
    print("kagglehub download path:", download_path)

    files = sorted(download_path.glob("*"))
    if not files:
        raise RuntimeError(f"No files found in {download_path}")

    for src in files:
        if src.is_file():
            dst = DATA_RAW_DIR / src.name
            if dst.exists() and dst.stat().st_size == src.stat().st_size:
                print(f"  [skip] {dst.name} already staged ({dst.stat().st_size / 1e6:.2f} MB)")
                continue
            shutil.copy2(src, dst)
            print(f"  [copy] {src.name} -> {dst} ({dst.stat().st_size / 1e6:.2f} MB)")

    print("Done! Staged files in:", DATA_RAW_DIR)


if __name__ == "__main__":
    main()
