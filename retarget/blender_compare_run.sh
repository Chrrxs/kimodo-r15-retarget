#!/usr/bin/env bash
# End-to-end Blender comparison video for one clip:
#   1. export R15 world-FK json next to the BVH (if missing)
#   2. render side-by-side frames with Blender (Workbench, headless CPU)
#   3. encode <clipdir>/motion_compare_blender.mp4 with the .venv's ffmpeg
#
# Usage: bash retarget/blender_compare_run.sh path/to/motion.bvh [WxH]
# Env:   BLENDER=/path/to/blender (default: blender on PATH)
set -euo pipefail

BVH="${1:?usage: blender_compare_run.sh <motion.bvh> [WxH]}"
SIZE="${2:-1280x720}"
BLENDER="${BLENDER:-blender}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$BVH" ]] || { echo "BVH not found: $BVH" >&2; exit 2; }
command -v "$BLENDER" >/dev/null || { echo "Blender not found: $BLENDER (set BLENDER=)" >&2; exit 2; }

CLIP_DIR="$(cd "$(dirname "$BVH")" && pwd)"
BVH="$CLIP_DIR/$(basename "$BVH")"
FK="$CLIP_DIR/r15_fk.json"
OUT="$CLIP_DIR/motion_compare_blender.mp4"

if [[ ! -f "$FK" ]]; then
    echo "Exporting R15 FK json -> $FK"
    # The Blender scene renders the classic blocky rig meshes, so use that
    # rig's joint geometry for the FK.
    python3 "$REPO/retarget/export_fk_json.py" "$BVH" --output "$FK" \
        --geometry "$REPO/retarget/r15_rig_geometry_kimodo.json"
fi

FRAMES_DIR="$(mktemp -d /tmp/blender_compare.XXXXXX)"
trap 'rm -rf "$FRAMES_DIR"' EXIT

"$BLENDER" -b --factory-startup -noaudio \
    --python "$REPO/retarget/blender_compare.py" -- \
    --bvh "$BVH" --fk "$FK" --output-dir "$FRAMES_DIR" --size "$SIZE"

# ffmpeg: project venv's imageio-ffmpeg -> any importable imageio-ffmpeg -> PATH
FFMPEG="$("$REPO/.venv/bin/python" -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null \
    || python3 -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null \
    || command -v ffmpeg)" || { echo "ffmpeg not found (pip install imageio-ffmpeg)" >&2; exit 2; }
FPS="$(python3 -c "import json; print(int(round(json.load(open('$FK'))['fps'])))")"

# NOTE: the bundled imageio-ffmpeg build has no drawtext filter; labels are
# baked into the frames as Blender text objects instead.
"$FFMPEG" -y -framerate "$FPS" -i "$FRAMES_DIR/frame_%04d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 -r "$FPS" "$OUT"

echo "Wrote $OUT"
