"""Download the pinned Tiny Shakespeare text and verify its SHA-256."""
from pathlib import Path
from urllib.request import urlopen
import hashlib

HERE = Path(__file__).resolve().parent
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EXPECTED = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
p = HERE / "data" / "shakespeare.txt"
p.parent.mkdir(exist_ok=True)
if not p.exists():
    data = urlopen(URL, timeout=60).read()
    if hashlib.sha256(data).hexdigest() != EXPECTED:
        raise ValueError("Downloaded dataset checksum mismatch; no file written")
    p.write_bytes(data)
if hashlib.sha256(p.read_bytes()).hexdigest() != EXPECTED:
    raise ValueError("Existing dataset checksum mismatch")
print("Dataset verified:", p)
