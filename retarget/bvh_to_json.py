#!/usr/bin/env python3
"""Dump a BVH's world-space joint positions per frame as compact JSON.

Runs the same parser/FK conventions as the retargeter (retarget/bvh.py +
math3d.py), so the output is exactly the motion the retargeter consumed.
Useful for visual QA of the source motion outside Roblox.

Usage: python3 retarget/bvh_to_json.py <input.bvh> [--output out.json] [--stride N]
Output JSON: {"name", "fps", "joints": [names...], "parents": [idx...],
              "frames": [[x,y,z per joint, cm, y-up]...]}
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bvh as bvh_mod  # noqa: E402
import math3d  # noqa: E402


def world_positions(anim, frame):
    """FK: world position of every joint for one frame, in BVH order."""
    pos = {}
    rot = {}
    out = []
    for joint in anim.joints:  # topological order (parents first)
        q_local = anim.joint_rotation_quat(joint.name, frame)
        t_local = anim.joint_translation(joint.name, frame)
        if joint.parent is None:
            pos[joint.name] = t_local
            rot[joint.name] = q_local
        else:
            p_q = rot[joint.parent.name]
            pos[joint.name] = math3d.vec_add(
                pos[joint.parent.name], math3d.quat_rotate(p_q, t_local)
            )
            rot[joint.name] = math3d.quat_mul(p_q, q_local)
        out.append(pos[joint.name])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth frame")
    args = ap.parse_args()

    anim = bvh_mod.Bvh.from_file(args.input)
    joints = [j.name for j in anim.joints]
    index = {n: i for i, n in enumerate(joints)}
    parents = [index[j.parent.name] if j.parent else -1 for j in anim.joints]

    frames = []
    for f in range(0, anim.nframes, max(1, args.stride)):
        flat = []
        for p in world_positions(anim, f):
            flat.extend(round(v, 1) for v in p)  # 1mm precision is plenty
        frames.append(flat)

    doc = {
        "name": Path(args.input).stem,
        "fps": (1.0 / anim.frame_time) / max(1, args.stride),
        "joints": joints,
        "parents": parents,
        "frames": frames,
    }
    out = args.output or (str(Path(args.input).with_suffix("")) + "_joints.json")
    with open(out, "w") as fh:
        json.dump(doc, fh, separators=(",", ":"))
    print(f"wrote {out}: {len(frames)} frames, {len(joints)} joints")


if __name__ == "__main__":
    main()
