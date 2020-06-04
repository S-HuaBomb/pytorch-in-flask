#!/bin/bash

# shellcheck disable=SC2125
FILES=pictures/*
OUTDIR=pictures/

for f in $FILES
do
  filename="${f##*/}"
  echo "Generating thumbnail for $filename..."
  convert "$f" -thumbnail 150x150^ -gravity center -extent 150x150 "$OUTDIR$filename"
done
