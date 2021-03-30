#!/bin/bash

DIR="$( dirname "${BASH_SOURCE[0]}" )"
cd $DIR

FILE=cvpr2021_results.zip
ID=1r8e5DS4e6rz-v0OQffgK_JoJBXCEBZq3
CHECKSUM=6b1e883d2c134ffe6228fe727d90650a

if [ -f $FILE ]; then
  echo "File already exists. Checking md5..."
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct. No need to download."
    exit 0
  else
    echo "Checksum is incorrect. Need to download again."
  fi
fi

echo "Downloading example results (2.5G)..."

wget --no-check-certificate -r "https://drive.google.com/uc?export=download&id=$ID" -O $FILE

echo "Unzipping..."

unzip $FILE

echo "Done."
