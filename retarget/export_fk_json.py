#!/usr/bin/env python3
"""Export per-frame world CFrames of the retargeted R15 rig as JSON.

Feeds the Blender comparison renderer (blender_compare.py): the R15 side is
reconstructed with the Animator's joint formula (see compare_video.fk_r15)
and rotated 180 deg about Y so the character faces +Z like the SOMA source.

Output: {"fps", "parts": {name: size_studs}, "frames": [{part: [px,py,pz,
qw,qx,qy,qz]}...]}  — positions in studs, Roblox axes (y-up, faces +Z after
the flip), quaternions wxyz.

Usage: python3 retarget/export_fk_json.py <input.bvh> --output <out.json>
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bvh as bvh_mod  # noqa: E402
import bvh_to_r15  # noqa: E402
import math3d  # noqa: E402
from compare_video import GEOMETRY_PATH, fk_r15, rest_floor_offset  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", required=True)
    ap.add_argument("--forward-axis", default="z", choices=["z", "-z", "x", "-x"])
    ap.add_argument("--scale-studs-per-meter", type=float, default=3.571)
    ap.add_argument("--geometry", default=str(GEOMETRY_PATH),
                    help="rig geometry JSON (default: generic rig; pass "
                         "r15_rig_geometry_kimodo.json for the Kimodo rig)")
    args = ap.parse_args()

    anim = bvh_mod.Bvh.from_file(args.input)
    frames, fps = bvh_to_r15.retarget_animation(
        anim, forward_axis=args.forward_axis, studs_per_meter=args.scale_studs_per_meter
    )
    geometry = json.loads(Path(args.geometry).read_text())
    hrp_y = rest_floor_offset(geometry)
    flip_q = math3d.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi)

    out_frames = []
    for f in range(anim.nframes):
        cfs = fk_r15(frames[f], geometry, ((0.0, hrp_y, 0.0), math3d.QUAT_IDENTITY))
        row = {}
        for part, (pos, q) in cfs.items():
            if part == "HumanoidRootPart":
                continue
            fp = math3d.quat_rotate(flip_q, pos)
            fq = math3d.quat_mul(flip_q, q)
            row[part] = [round(v, 5) for v in (*fp, *fq)]
        out_frames.append(row)

    doc = {
        "fps": fps,
        "studs_per_meter": args.scale_studs_per_meter,
        "parts": {k: v for k, v in geometry["parts"].items() if k != "HumanoidRootPart"},
        "frames": out_frames,
    }
    Path(args.output).write_text(json.dumps(doc, separators=(",", ":")))
    print(f"wrote {args.output}: {len(out_frames)} frames @ {fps:.1f} fps")


if __name__ == "__main__":
    main()
