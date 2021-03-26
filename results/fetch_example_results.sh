#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=example_results.tar.gz
ID=1b_rS0EHoji9a41c17v2hMQKMnMoM1oC4

if [ -f $FILE ]; then
  echo "File already exists."
  exit 0
fi

echo "Downloading example results (3M)..."

wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=$ID" -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
