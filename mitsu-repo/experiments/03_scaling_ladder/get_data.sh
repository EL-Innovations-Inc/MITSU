#!/bin/bash
for f in train valid; do
  curl -sSL -o wiki_$f.txt https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/$f.txt
done
