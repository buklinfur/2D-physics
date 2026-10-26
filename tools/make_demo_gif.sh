#!/usr/bin/env bash
# Render the README demo animation from frames written by `executables --dump-frames`.
#
# Usage: tools/make_demo_gif.sh FRAMES_DIR OUTPUT.(gif|mp4) [--color] [--width PX] [--fps N]
#   --color   map speed to the "turbo" colormap instead of the app's grayscale
#   --width   output width in pixels (default 800; height keeps the aspect ratio)
#   --fps     playback rate (default 30)
set -euo pipefail

if [ $# -lt 2 ]; then sed -n '2,8p' "$0"; exit 2; fi
FRAMES=$1; OUTPUT=$2; shift 2
COLOR=0; WIDTH=800; FPS=30
while [ $# -gt 0 ]; do
    case "$1" in
        --color) COLOR=1 ;;
        --width) WIDTH=$2; shift ;;
        --fps)   FPS=$2; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

FILTER="scale=${WIDTH}:-2:flags=lanczos"
[ "$COLOR" = 1 ] && FILTER="format=gray,pseudocolor=preset=turbo,${FILTER}"

case "$OUTPUT" in
    *.gif)
        # Two-pass palette for a clean, small GIF.
        ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%05d.pgm" \
            -vf "${FILTER},split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4" \
            -loop 0 "$OUTPUT" ;;
    *.mp4)
        ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%05d.pgm" \
            -vf "${FILTER},format=yuv420p" -c:v libx264 -crf 20 -preset veryfast -movflags +faststart "$OUTPUT" ;;
    *) echo "output must end in .gif or .mp4" >&2; exit 2 ;;
esac
echo "Wrote $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
