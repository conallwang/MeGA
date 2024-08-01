#!/bin/bash

IMG_FOLDER=$1
ROOT=$(dirname "$IMG_FOLDER")

ffmpeg -i $IMG_FOLDER/%05d.png -c:v libx264 -r 30 $ROOT/output.mp4