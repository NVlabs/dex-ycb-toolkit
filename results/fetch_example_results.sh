#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=example_results.tar.gz
ID=1krLY5xkCSmwygVqcqW6t9o5dYAEahGEW

if [ -f $FILE ]; then
  echo "File already exists."
  exit 0
fi

echo "Downloading example results (3M)..."

wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=$ID" -O $FILE

echo "Unzipping..."

tar zxvf $FILE

echo "Done."
