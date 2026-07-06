#!/usr/bin/env python3
"""BVH (SOMA somaskel77) -> Roblox R15 KeyframeSequence (.rbxmx) retargeter.

Usage:
    python3 retarget/bvh_to_r15.py input.bvh --output out.rbxmx \
        [--rig r15] [--scale-studs-per-meter 3.571] [--fps N] \
        [--keyframe-reduction-epsilon 0.5] [--loop] [--priority Action] \
        [--forward-axis z|-z|x|-x] [--name AnimName]

Pipeline:
    parse BVH -> per-frame local rotations per SOMA joint -> compose collapsed
    chains -> axis conversion + rest-pose correction -> per-Motor6D Transform
    CFrames -> optional resampling -> optional keyframe reduction -> .rbxmx
    (+ a <out>.meta.json sidecar with stats).

Coordinate conventions:
    BVH/kimodo is right-handed Y-up, character facing +Z (configurable via
    --forward-axis).  Roblox is right-handed Y-up, characters face -Z.  The
    conversion C is a pure rotation about +Y chosen so that the BVH forward
    axis maps onto Roblox -Z; it is applied by conjugation to rotations
    (C q C^-1) and directly to translations.
"""

import argparse
import json
import math
import os
import sys

try:
    from . import bvh as bvh_mod
    from . import math3d, r15, rbxmx
except ImportError:
    import bvh as bvh_mod
    import math3d
    import r15
    import rbxmx

# Rotation about +Y (degrees) mapping each BVH forward axis onto Roblox -Z.
_FORWARD_AXIS_YAW_DEG = {"z": 180.0, "-z": 0.0, "x": 90.0, "-x": -90.0}

#: Translation tolerance (studs) used alongside the angular epsilon during
#: keyframe reduction.
_REDUCTION_POS_EPSILON_STUDS = 0.01


def conversion_quat(forward_axis):
    """Axis-conversion rotation C (BVH -> Roblox axes) as a quaternion."""
    try:
        yaw = _FORWARD_AXIS_YAW_DEG[forward_axis]
    except KeyError:
        raise ValueError("--forward-axis must be one of %s" % ", ".join(_FORWARD_AXIS_YAW_DEG))
    return math3d.quat_from_axis_angle((0.0, 1.0, 0.0), math.radians(yaw))


def validate_skeleton(anim):
    missing = [j for j in r15.REQUIRED_SOMA_JOINTS if j not in anim.joint_map]
    if missing:
        raise ValueError(
            "input BVH is missing required SOMA joints: %s (is this a kimodo "
            "somaskel77 export?)" % ", ".join(missing)
        )


def retarget_frame(anim, frame, conv_q, corrections, studs_per_cm, hips_rest_cm):
    """Retarget one BVH frame -> dict part -> ((x,y,z) studs, quaternion)."""
    conv_q_inv = math3d.quat_conj(conv_q)

    # --- Rotations ---------------------------------------------------------
    transforms = {}
    for part in r15.PART_ORDER:
        if part == "HumanoidRootPart":
            continue
        spec = r15.MAPPING[part]
        # Composed local rotation of the collapsed SOMA chain.  Valid as a
        # single-joint rotation because every chain joint has an identity
        # local rest rotation in the standard T-pose export.
        l_src = math3d.QUAT_IDENTITY
        for joint_name in spec["chain"]:
            l_src = math3d.quat_mul(l_src, anim.joint_rotation_quat(joint_name, frame))
        # Change of basis into Roblox axes, then rest-pose correction:
        # T = D_parent^-1 * (C l C^-1) * D_part  (see retarget/r15.py).
        l_rbx = math3d.quat_mul(math3d.quat_mul(conv_q, l_src), conv_q_inv)
        d_parent = corrections[r15.POSE_PARENT[part]]
        q = math3d.quat_mul(math3d.quat_mul(math3d.quat_conj(d_parent), l_rbx), corrections[part])
        transforms[part] = ((0.0, 0.0, 0.0), math3d.quat_normalize(q))

    # --- Root motion -------------------------------------------------------
    # World hips position = Root position channels + Root rotation applied to
    # the Hips position channels (position channels are the full local
    # translation in kimodo exports).  The delta from the rest-pose hips
    # position rides on the LowerTorso Pose (the Root Motor6D transform):
    # Roblox's Animator ignores translation on the HumanoidRootPart Pose, so
    # the HRP Pose stays identity and the whole body moves relative to the
    # stationary HumanoidRootPart (verified empirically in Studio).
    p_root = anim.joint_translation("Root", frame)
    q_root = anim.joint_rotation_quat("Root", frame)
    p_hips = anim.joint_translation("Hips", frame)
    hips_world_cm = math3d.vec_add(p_root, math3d.quat_rotate(q_root, p_hips))
    delta_cm = math3d.vec_sub(hips_world_cm, hips_rest_cm)
    delta_studs = math3d.vec_scale(math3d.quat_rotate(conv_q, delta_cm), studs_per_cm)
    transforms["LowerTorso"] = (delta_studs, transforms["LowerTorso"][1])
    transforms["HumanoidRootPart"] = ((0.0, 0.0, 0.0), math3d.QUAT_IDENTITY)
    return transforms


def retarget_animation(anim, forward_axis="z", studs_per_meter=3.571):
    """Retarget all frames.  Returns (frames, fps) where each frame is a dict
    part -> (position_studs, quaternion)."""
    validate_skeleton(anim)
    conv_q = conversion_quat(forward_axis)
    corrections = r15.compute_rest_corrections(conv_q)
    studs_per_cm = studs_per_meter / 100.0
    hips_rest_cm = anim.rest_world_position("Hips")
    frames = [
        retarget_frame(anim, f, conv_q, corrections, studs_per_cm, hips_rest_cm)
        for f in range(anim.nframes)
    ]
    return frames, anim.fps


def resample(frames, src_fps, dst_fps):
    """Resample retargeted frames to dst_fps with slerp/lerp interpolation."""
    if not frames or abs(src_fps - dst_fps) < 1e-9:
        return frames, src_fps
    duration = (len(frames) - 1) / src_fps
    count = max(1, int(round(duration * dst_fps)) + 1)
    out = []
    for j in range(count):
        t = min(j / dst_fps, duration) * src_fps  # position in source frames
        i0 = min(int(math.floor(t)), len(frames) - 1)
        i1 = min(i0 + 1, len(frames) - 1)
        u = t - i0
        f0, f1 = frames[i0], frames[i1]
        blended = {}
        for part in r15.PART_ORDER:
            p0, q0 = f0[part]
            p1, q1 = f1[part]
            blended[part] = (math3d.lerp_vec(p0, p1, u), math3d.quat_slerp(q0, q1, u))
        out.append(blended)
    return out, dst_fps


def _frame_error(frame, f0, f1, u):
    """Max deviation of `frame` vs interpolating f0->f1 at fraction u.

    Returns (max_rotation_error_deg, max_position_error_studs)."""
    max_rot = 0.0
    max_pos = 0.0
    for part in r15.PART_ORDER:
        p, q = frame[part]
        p0, q0 = f0[part]
        p1, q1 = f1[part]
        qi = math3d.quat_slerp(q0, q1, u)
        max_rot = max(max_rot, math.degrees(math3d.quat_angle_between(qi, q)))
        pi = math3d.lerp_vec(p0, p1, u)
        max_pos = max(max_pos, math3d.vec_length(math3d.vec_sub(p, pi)))
    return max_rot, max_pos


def reduce_keyframes(frames, epsilon_deg, pos_epsilon=_REDUCTION_POS_EPSILON_STUDS):
    """Greedy keyframe reduction: keep a frame only when linear interpolation
    between kept neighbours would deviate by more than epsilon.

    Returns list of kept frame indices (always includes first and last).
    """
    n = len(frames)
    if n <= 2 or epsilon_deg <= 0.0:
        return list(range(n))
    kept = [0]
    i = 0
    while i < n - 1:
        # Find the farthest j such that all frames in (i, j) are within
        # tolerance of the i->j interpolation.
        best = i + 1
        for j in range(i + 2, n):
            ok = True
            for k in range(i + 1, j):
                u = (k - i) / (j - i)
                rot_err, pos_err = _frame_error(frames[k], frames[i], frames[j], u)
                if rot_err > epsilon_deg or pos_err > pos_epsilon:
                    ok = False
                    break
            if not ok:
                break
            best = j
        kept.append(best)
        i = best
    return kept


def convert(
    input_path,
    output_path,
    name=None,
    rig="r15",
    studs_per_meter=3.571,
    fps=None,
    reduction_epsilon_deg=0.5,
    loop=False,
    priority="Action",
    forward_axis="z",
):
    """Full pipeline.  Returns the metadata dict (also written as sidecar)."""
    if rig != "r15":
        raise ValueError("only --rig r15 is supported")
    anim = bvh_mod.Bvh.from_file(input_path)
    frames, src_fps = retarget_animation(anim, forward_axis, studs_per_meter)
    eff_fps = src_fps
    if fps is not None:
        frames, eff_fps = resample(frames, src_fps, float(fps))

    kept = reduce_keyframes(frames, reduction_epsilon_deg)
    keyframes = [(idx / eff_fps, frames[idx]) for idx in kept]

    if name is None:
        name = os.path.splitext(os.path.basename(input_path))[0]
    rbxmx.write_rbxmx(output_path, name, keyframes, loop=loop, priority=priority)

    meta = {
        "input": os.path.abspath(input_path),
        "output": os.path.abspath(output_path),
        "name": name,
        "rig": rig,
        "source_fps": src_fps,
        "output_fps": eff_fps,
        "source_frames": anim.nframes,
        "retargeted_frames": len(frames),
        "keyframes_written": len(keyframes),
        "duration_seconds": (len(frames) - 1) / eff_fps if len(frames) > 1 else 0.0,
        "reduction": {
            "epsilon_deg": reduction_epsilon_deg,
            "position_epsilon_studs": _REDUCTION_POS_EPSILON_STUDS,
            "frames_in": len(frames),
            "frames_out": len(keyframes),
            "ratio": (len(keyframes) / len(frames)) if frames else 1.0,
        },
        "scale_studs_per_meter": studs_per_meter,
        "forward_axis": forward_axis,
        "loop": loop,
        "priority": priority,
        "joints_mapped": {part: r15.MAPPING[part]["chain"] for part in r15.MAPPING},
        "motor6d_of_part": r15.MOTOR6D_OF_PART,
    }
    meta_path = output_path + ".meta.json"
    with open(meta_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    return meta


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Retarget a kimodo/SOMA somaskel77 BVH onto a Roblox R15 "
        "rig and write a KeyframeSequence .rbxmx."
    )
    parser.add_argument("input", help="input .bvh file (somaskel77, standard T-pose rest)")
    parser.add_argument("--output", "-o", required=True, help="output .rbxmx path")
    parser.add_argument("--rig", default="r15", choices=["r15"], help="target rig (default: r15)")
    parser.add_argument(
        "--scale-studs-per-meter",
        type=float,
        default=3.571,
        help="root-motion scale in studs per meter (default: 3.571; a typical "
        "R15 avatar is ~5 studs tall vs ~1.7 m human)",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=None,
        help="resample to this frame rate (default: keep source rate)",
    )
    parser.add_argument(
        "--keyframe-reduction-epsilon",
        type=float,
        default=0.5,
        metavar="DEG",
        help="drop frames reproducible by linear interpolation within this "
        "many degrees (0 disables; default: 0.5)",
    )
    parser.add_argument("--loop", action="store_true", help="set KeyframeSequence.Loop = true")
    parser.add_argument(
        "--priority",
        default="Action",
        choices=sorted(rbxmx.ANIMATION_PRIORITY),
        help="Enum.AnimationPriority (default: Action)",
    )
    parser.add_argument(
        "--forward-axis",
        default="z",
        choices=sorted(_FORWARD_AXIS_YAW_DEG),
        help="axis the BVH character faces (default: z, per the somaskel77 "
        "standard T-pose; mapped onto Roblox -Z)",
    )
    parser.add_argument("--name", default=None, help="animation name (default: input basename)")
    args = parser.parse_args(argv)

    meta = convert(
        args.input,
        args.output,
        name=args.name,
        rig=args.rig,
        studs_per_meter=args.scale_studs_per_meter,
        fps=args.fps,
        reduction_epsilon_deg=args.keyframe_reduction_epsilon,
        loop=args.loop,
        priority=args.priority,
        forward_axis=args.forward_axis,
    )
    print(
        "wrote %s: %d keyframes (%d source frames, %.2fs @ %.3g fps)"
        % (
            meta["output"],
            meta["keyframes_written"],
            meta["source_frames"],
            meta["duration_seconds"],
            meta["output_fps"],
        )
    )
    print("metadata: %s.meta.json" % meta["output"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
