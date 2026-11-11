"""Verify the SHA256SUMS file of the extracted source ZIP (stdlib only)."""
from hashlib import sha256
from pathlib import Path
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = root / "SHA256SUMS"
    if not manifest.exists():
        raise SystemExit("SHA256SUMS is supplied in the complete ZIP, not the wheel/sdist.")
    failures = []
    count = 0
    for line in manifest.read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            failures.append(relative)
            continue
        count += 1
        if sha256(path.read_bytes()).hexdigest() != expected:
            failures.append(relative)
    if failures:
        print("Missing or modified: " + ", ".join(failures), file=sys.stderr)
        return 1
    print(f"PASS: {count} release file checksums match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
