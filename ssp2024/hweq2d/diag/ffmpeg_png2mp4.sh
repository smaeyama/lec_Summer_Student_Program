#!/bin/sh

# 2024/06/26
#ffmpeg -r 30 -i png_$1/$1_t%08d.png -pix_fmt yuv420p -r 60 anim_$1.mp4

# 2024/08/21
ffmpeg -r 30 -pattern_type glob -i "png_$1/$1_t*.png" -pix_fmt yuv420p -c:v libx264 -crf 23 -r 60 anim_$1.mp4
