#!/bin/sh

ffmpeg -r 30 -i png_$1/$1_t%08d.png -pix_fmt yuv420p -r 60 anim_$1.mp4
