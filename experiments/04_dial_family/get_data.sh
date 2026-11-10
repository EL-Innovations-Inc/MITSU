#!/usr/bin/env bash
# Download the three datasets into ./data (about 42 MB). Checksums: data/SHA256.txt
set -euo pipefail
cd "$(dirname "$0")/data"
FILES="train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz"
for f in $FILES; do
  [ -f "mnist-$f" ]   || curl -fsSL -o "mnist-$f"   "https://raw.githubusercontent.com/fgnt/mnist/master/$f"
  [ -f "fashion-$f" ] || curl -fsSL -o "fashion-$f" "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f"
done
[ -f shakespeare.txt ] || curl -fsSL -o shakespeare.txt "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
sha256sum -c SHA256.txt
