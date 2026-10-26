"""Download explicitly selected raw experimental files and verify published MD5 values."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("outputs/experimental_raw"))
    parser.add_argument(
        "--list", action="store_true", help="List sources without downloading"
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sources = json.loads((root / "configs/experimental_sources.json").read_text())[
        "raw_files"
    ]
    if args.list:
        print(json.dumps(sources, indent=2))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    for row in sources:
        target = args.output / row["name"]
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".partial")
            with (
                urlopen(row["url"], timeout=120) as response,
                temporary.open("wb") as stream,
            ):
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            temporary.replace(target)
        digest = hashlib.md5()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["md5"]:
            raise RuntimeError(f"Published checksum mismatch: {row['name']}")
        print(f"Verified {row['name']}")


if __name__ == "__main__":
    main()
