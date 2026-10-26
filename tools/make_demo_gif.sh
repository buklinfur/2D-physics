#!/usr/bin/env bash
# Render the README demo animation from frames written by `executables --dump-frames`.
#
# Usage: tools/make_demo_gif.sh FRAMES_DIR OUTPUT.(gif|mp4) [--color] [--width PX] [--fps N] [--take K]
#   --color   map speed to the "turbo" colormap instead of the app's grayscale
#   --width   output width in pixels (default 800; height keeps the aspect ratio)
#   --fps     playback rate (default 30)
#   --take    use every K-th frame (default 1); K=2 roughly halves the GIF size
set -euo pipefail

if [ $# -lt 2 ]; then sed -n '2,9p' "$0"; exit 2; fi
FRAMES=$1; OUTPUT=$2; shift 2
COLOR=0; WIDTH=800; FPS=30; TAKE=1
while [ $# -gt 0 ]; do
    case "$1" in
        --color) COLOR=1 ;;
        --width) WIDTH=$2; shift ;;
        --fps)   FPS=$2; shift ;;
        --take)  TAKE=$2; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
    shift
done

FILTER="select='not(mod(n\\,${TAKE}))',setpts=N/(${FPS}*TB)"
# pseudocolor needs a multi-plane format: on a gray input it leaves the image gray.
[ "$COLOR" = 1 ] && FILTER="${FILTER},format=yuv444p,pseudocolor=preset=turbo"
FILTER="${FILTER},scale=${WIDTH}:-2:flags=lanczos"

case "$OUTPUT" in
    *.gif)
        # Two-pass palette for a clean, small GIF.
        ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%05d.pgm" \
            -vf "${FILTER},split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=none:diff_mode=rectangle" \
            -r "$FPS" -loop 0 "$OUTPUT" ;;
    *.mp4)
        ffmpeg -y -loglevel error -framerate "$FPS" -i "$FRAMES/frame_%05d.pgm" \
            -vf "${FILTER},format=yuv420p" -r "$FPS" -c:v libx264 -crf 20 -preset veryfast -movflags +faststart "$OUTPUT" ;;
    *) echo "output must end in .gif or .mp4" >&2; exit 2 ;;
esac
echo "Wrote $OUTPUT ($(du -h "$OUTPUT" | cut -f1))"
