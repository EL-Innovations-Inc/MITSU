#!/bin/bash
mkdir -p data && cd data
for f in train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz; do
  curl -sSL -o mnist-$f   https://raw.githubusercontent.com/fgnt/mnist/master/$f
  curl -sSL -o fashion-$f https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/$f
done
