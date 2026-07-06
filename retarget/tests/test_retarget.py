#!/usr/bin/env python3
"""Tests for the BVH -> R15 retargeter.

Plain-python test runner: no pytest needed.

    python3 retarget/tests/test_retarget.py
"""

import math
import os
import sys
import tempfile
import xml.etree.ElementTree as ET

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(_HERE))  # retarget/ on sys.path

import bvh as bvh_mod  # noqa: E402
import bvh_to_r15  # noqa: E402
import math3d  # noqa: E402
import r15  # noqa: E402
import rbxmx  # noqa: E402

TPOSE_PATH = os.path.join(_HERE, "data", "somaskel77_standard_tpose.bvh")

DEG = math.degrees


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def vec_approx(a, b, tol=1e-6):
    return all(abs(x - y) <= tol for x, y in zip(a, b))


def quat_close(a, b, tol_deg=1e-4):
    return DEG(math3d.quat_angle_between(a, b)) <= tol_deg


# ---------------------------------------------------------------------------
# BVH parsing
# ---------------------------------------------------------------------------


def test_bvh_parsing():
    anim = bvh_mod.Bvh.from_file(TPOSE_PATH)
    assert anim.root.name == "Root"
    assert anim.nframes == 1
    assert approx(anim.frame_time, 1.0 / 30.0, 1e-9)
    assert approx(anim.fps, 30.0, 1e-6)

    # 77 SOMA joints + the "Root" wrapper.
    assert len(anim.joints) == 78, len(anim.joints)

    # Channel layout: Root and Hips have 6 channels, everything else 3.
    assert anim.joint_map["Root"].channels == [
        "Xposition", "Yposition", "Zposition", "Zrotation", "Yrotation", "Xrotation",
    ]
    assert anim.joint_map["Hips"].channels == anim.joint_map["Root"].channels
    assert anim.joint_map["Spine1"].channels == ["Zrotation", "Yrotation", "Xrotation"]
    assert anim.joint_map["Spine1"].rotation_order == "ZYX"

    # Offsets straight from the file.
    assert vec_approx(anim.joint_map["Hips"].offset, (0.0, 100.0, 0.0))
    assert vec_approx(anim.joint_map["LeftForeArm"].offset, (28.739307, 0.0, -0.002588))

    # All mapped joints exist.
    for j in r15.REQUIRED_SOMA_JOINTS:
        assert j in anim.joint_map, j

    # Rest world positions accumulate offsets.
    assert vec_approx(anim.rest_world_position("Hips"), (0.0, 100.0, 0.0))
    knee = anim.rest_world_position("LeftShin")
    assert approx(knee[1], 100.0 - 8.434526 - 43.221752, 1e-4), knee

    # T-pose frame: all rotations zero, positions match offsets.
    for j in r15.REQUIRED_SOMA_JOINTS:
        assert quat_close(anim.joint_rotation_quat(j, 0), math3d.QUAT_IDENTITY)
    assert vec_approx(anim.joint_translation("Hips", 0), (0.0, 100.0, 0.0))
    assert vec_approx(anim.joint_translation("Root", 0), (0.0, 0.0, 0.0))
    # Joint without position channels falls back to its offset.
    assert vec_approx(anim.joint_translation("LeftForeArm", 0), (28.739307, 0.0, -0.002588))


# ---------------------------------------------------------------------------
# Quaternion / math round-trips
# ---------------------------------------------------------------------------


def test_quat_math():
    # Euler (BVH ZYX intrinsic) -> quat -> matrix -> quat round trip.
    cases = [
        ("ZYX", (30.0, -45.0, 60.0)),
        ("ZYX", (0.0, 0.0, 90.0)),
        ("ZYX", (180.0, 0.0, 0.0)),
        ("ZYX", (-170.0, 85.0, 12.5)),
        ("XYZ", (10.0, 20.0, 30.0)),
    ]
    for order, deg in cases:
        q = math3d.quat_from_bvh_euler(order, deg)
        m = math3d.quat_to_mat3(q)
        q2 = math3d.mat3_to_quat(m)
        assert quat_close(q, q2, 1e-6), (order, deg)

    # ZYX intrinsic == Rz @ Ry @ Rx: check against composed axis rotations.
    qz = math3d.quat_from_axis_angle((0, 0, 1), math.radians(30))
    qy = math3d.quat_from_axis_angle((0, 1, 0), math.radians(-45))
    qx = math3d.quat_from_axis_angle((1, 0, 0), math.radians(60))
    q_ref = math3d.quat_mul(math3d.quat_mul(qz, qy), qx)
    assert quat_close(math3d.quat_from_bvh_euler("ZYX", (30, -45, 60)), q_ref, 1e-9)

    # Rotation sanity: +90 deg about +Y takes +X to -Z (right-hand rule).
    q = math3d.quat_from_axis_angle((0, 1, 0), math.radians(90))
    assert vec_approx(math3d.quat_rotate(q, (1, 0, 0)), (0, 0, -1), 1e-9)

    # quat_rotate agrees with the matrix form.
    v = (0.3, -1.2, 2.5)
    m = math3d.quat_to_mat3(q_ref)
    mv = tuple(sum(m[r][c] * v[c] for c in range(3)) for r in range(3))
    assert vec_approx(math3d.quat_rotate(q_ref, v), mv, 1e-9)

    # quat_between: takes u to v; identity and antiparallel edge cases.
    u, v = (0.0, -1.0, 0.0), (-1.0, 0.0, 0.0)
    q = math3d.quat_between(u, v)
    assert vec_approx(math3d.quat_rotate(q, u), v, 1e-9)
    assert quat_close(math3d.quat_between(u, u), math3d.QUAT_IDENTITY, 1e-6)
    q180 = math3d.quat_between((0, 1, 0), (0, -1, 0))
    assert vec_approx(math3d.quat_rotate(q180, (0, 1, 0)), (0, -1, 0), 1e-9)
    assert approx(DEG(math3d.quat_angle(q180)), 180.0, 1e-6)

    # slerp endpoints and midpoint angle.
    a = math3d.QUAT_IDENTITY
    b = math3d.quat_from_axis_angle((0, 0, 1), math.radians(90))
    assert quat_close(math3d.quat_slerp(a, b, 0.0), a, 1e-6)
    assert quat_close(math3d.quat_slerp(a, b, 1.0), b, 1e-6)
    mid = math3d.quat_slerp(a, b, 0.5)
    assert approx(DEG(math3d.quat_angle(mid)), 45.0, 1e-6)


# ---------------------------------------------------------------------------
# Bone-direction declarations vs the reference T-pose geometry
# ---------------------------------------------------------------------------


def test_declared_bone_directions_match_tpose():
    """The declared src_dir vectors must agree with the actual rest geometry.

    Measured as the direction from a representative chain joint to its child;
    feet use a looser threshold because the ankle->toe vector intentionally
    dips below the declared horizontal 'forward'."""
    anim = bvh_mod.Bvh.from_file(TPOSE_PATH)
    probes = {  # part -> (from_joint, to_joint, min_cosine)
        "LowerTorso": ("Hips", "Spine1", 0.98),
        "UpperTorso": ("Spine1", "Neck1", 0.98),
        # The neck segment itself (not through HeadEnd): SOMA's rest neck
        # leans ~17 deg forward and src_dir must match it tightly, otherwise
        # heads pitch up by the difference during motion.
        "Head": ("Neck1", "Head", 0.999),
        "LeftUpperArm": ("LeftArm", "LeftForeArm", 0.99),
        "LeftLowerArm": ("LeftForeArm", "LeftHand", 0.99),
        "RightUpperArm": ("RightArm", "RightForeArm", 0.99),
        "RightLowerArm": ("RightForeArm", "RightHand", 0.99),
        "LeftUpperLeg": ("LeftLeg", "LeftShin", 0.98),
        "LeftLowerLeg": ("LeftShin", "LeftFoot", 0.98),
        "RightUpperLeg": ("RightLeg", "RightShin", 0.98),
        "RightLowerLeg": ("RightShin", "RightFoot", 0.98),
        "LeftFoot": ("LeftFoot", "LeftToeBase", 0.85),
        "RightFoot": ("RightFoot", "RightToeBase", 0.85),
    }
    for part, (a, b, min_cos) in probes.items():
        measured = math3d.vec_normalize(
            math3d.vec_sub(anim.rest_world_position(b), anim.rest_world_position(a))
        )
        declared = math3d.vec_normalize(r15.MAPPING[part]["src_dir"])
        cos = math3d.vec_dot(measured, declared)
        assert cos >= min_cos, (part, measured, declared, cos)


# ---------------------------------------------------------------------------
# Rest corrections: shoulders get +/-90 deg roll, everything else ~identity
# ---------------------------------------------------------------------------


def test_rest_corrections():
    conv_q = bvh_to_r15.conversion_quat("z")
    corrections = r15.compute_rest_corrections(conv_q)

    # Left shoulder: -90 deg about Z (arm-down maps to character's left, -X).
    d = corrections["LeftUpperArm"]
    assert approx(DEG(math3d.quat_angle(d)), 90.0, 1e-6), d
    assert vec_approx(math3d.quat_rotate(d, (0, -1, 0)), (-1, 0, 0), 1e-9)
    expected_l = math3d.quat_from_axis_angle((0, 0, 1), math.radians(-90))
    assert quat_close(d, expected_l, 1e-6)

    # Right shoulder: +90 deg about Z.
    d = corrections["RightUpperArm"]
    expected_r = math3d.quat_from_axis_angle((0, 0, 1), math.radians(90))
    assert quat_close(d, expected_r, 1e-6)

    # Elbow/wrist share the arm's correction (so their rest transforms cancel).
    assert quat_close(corrections["LeftLowerArm"], corrections["LeftUpperArm"], 1e-6)
    assert quat_close(corrections["LeftHand"], corrections["LeftUpperArm"], 1e-6)

    # Torso, legs, feet: identity (matching physical rest poses).
    for part in [
        "LowerTorso", "UpperTorso",
        "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
        "RightUpperLeg", "RightLowerLeg", "RightFoot",
    ]:
        assert quat_close(corrections[part], math3d.QUAT_IDENTITY, 1e-4), part

    # Head: SOMA's rest neck leans ~17.1 deg forward, so the correction is a
    # matching forward pitch (about -X in Roblox axes after z -> -z).
    d = corrections["Head"]
    neck_lean = math.atan2(r15._NECK_DIR[2], r15._NECK_DIR[1])  # ~17.1 deg
    assert approx(DEG(math3d.quat_angle(d)), DEG(neck_lean), 0.01), d
    expected_head = math3d.quat_from_axis_angle((1, 0, 0), -neck_lean)
    assert quat_close(d, expected_head, 0.01), d


# ---------------------------------------------------------------------------
# End to end: T-pose in -> valid rbxmx with self-consistent transforms
# ---------------------------------------------------------------------------


def _parse_pose_tree(pose_item):
    """Recursively parse a Pose Item -> {name: (translation, mat3, children)}."""
    props = pose_item.find("Properties")
    name = props.find("string[@name='Name']").text
    cf = props.find("CoordinateFrame[@name='CFrame']")
    vals = {c.tag: float(c.text) for c in cf}
    trans = (vals["X"], vals["Y"], vals["Z"])
    mat = (
        (vals["R00"], vals["R01"], vals["R02"]),
        (vals["R10"], vals["R11"], vals["R12"]),
        (vals["R20"], vals["R21"], vals["R22"]),
    )
    assert props.find("float[@name='Weight']").text == "1"
    children = {}
    for child in pose_item.findall("Item[@class='Pose']"):
        children.update(_parse_pose_tree(child))
    flat = {name: (trans, mat)}
    flat.update(children)
    # Verify nesting matches the R15 pose tree.
    for child in pose_item.findall("Item[@class='Pose']"):
        cname = child.find("Properties/string[@name='Name']").text
        assert r15.POSE_PARENT[cname] == name, (cname, name)
    return flat


def test_tpose_end_to_end():
    out_dir = tempfile.mkdtemp(prefix="retarget_test_")
    out_path = os.path.join(out_dir, "tpose.rbxmx")
    meta = bvh_to_r15.convert(TPOSE_PATH, out_path, name="tpose_test", loop=True,
                              priority="Idle")
    assert os.path.exists(out_path)
    assert os.path.exists(out_path + ".meta.json")
    assert meta["source_frames"] == 1
    assert meta["keyframes_written"] == 1

    tree = ET.parse(out_path)  # raises if not well-formed XML
    root = tree.getroot()
    assert root.tag == "roblox" and root.get("version") == "4"

    seq = root.find("Item[@class='KeyframeSequence']")
    assert seq is not None
    props = seq.find("Properties")
    assert props.find("string[@name='Name']").text == "tpose_test"
    assert props.find("bool[@name='Loop']").text == "true"
    assert props.find("token[@name='Priority']").text == str(rbxmx.ANIMATION_PRIORITY["Idle"])

    keyframes = seq.findall("Item[@class='Keyframe']")
    assert len(keyframes) == 1
    assert float(keyframes[0].find("Properties/float[@name='Time']").text) == 0.0

    hrp = keyframes[0].find("Item[@class='Pose']")
    poses = _parse_pose_tree(hrp)
    assert set(poses) == set(r15.PART_ORDER), set(r15.PART_ORDER) - set(poses)

    def pose_quat(part):
        return math3d.mat3_to_quat(poses[part][1])

    # HumanoidRootPart: identity (no root offset in the T-pose file).
    trans, _ = poses["HumanoidRootPart"]
    assert vec_approx(trans, (0, 0, 0), 1e-6)
    assert quat_close(pose_quat("HumanoidRootPart"), math3d.QUAT_IDENTITY, 1e-3)

    # Self-consistency: retargeting the rest pose must reproduce exactly the
    # rest corrections: shoulders ~= -/+90 deg roll, head ~= 17 deg forward
    # pitch (SOMA's leaned rest neck faithfully re-expressed on the R15 rig),
    # everything else identity.
    corrections = r15.compute_rest_corrections(bvh_to_r15.conversion_quat("z"))
    for part in r15.PART_ORDER:
        if part == "HumanoidRootPart":
            continue
        expected = (
            corrections[part]
            if part in ("LeftUpperArm", "RightUpperArm", "Head")
            else math3d.QUAT_IDENTITY
        )
        assert quat_close(pose_quat(part), expected, 1e-3), (
            part, pose_quat(part), expected,
        )
        # Motor6D transforms never carry translation (except HRP root motion).
        assert vec_approx(poses[part][0], (0, 0, 0), 1e-9), part

    q = pose_quat("LeftUpperArm")
    assert approx(DEG(math3d.quat_angle(q)), 90.0, 1e-3)
    assert quat_close(q, math3d.quat_from_axis_angle((0, 0, 1), math.radians(-90)), 1e-3)
    q = pose_quat("RightUpperArm")
    assert quat_close(q, math3d.quat_from_axis_angle((0, 0, 1), math.radians(90)), 1e-3)


# ---------------------------------------------------------------------------
# Semantic checks on an animated (mutated) pose
# ---------------------------------------------------------------------------


def _set_channels(anim, frame, joint_name, channel_values):
    """channel_values: dict channel name -> value (degrees / cm)."""
    joint = anim.joint_map[joint_name]
    for k, ch in enumerate(joint.channels):
        if ch in channel_values:
            anim.frames[frame][joint.channel_start + k] = channel_values[ch]


def test_elbow_bend_semantics():
    """Bend the SOMA left elbow so the forearm points character-forward; the
    R15 LeftLowerArm transform must be +90 deg about +X (forearm from
    arms-down to pointing Roblox-forward, -Z)."""
    anim = bvh_mod.Bvh.from_file(TPOSE_PATH)
    # Left forearm dir is +X; rotating -90 deg about +Y takes +X to +Z
    # (character forward in BVH axes).
    _set_channels(anim, 0, "LeftForeArm", {"Yrotation": -90.0})
    frames, _ = bvh_to_r15.retarget_animation(anim, forward_axis="z")
    _, q = frames[0]["LeftLowerArm"]
    expected = math3d.quat_from_axis_angle((1, 0, 0), math.radians(90))
    assert quat_close(q, expected, 1e-3), q
    # Check the resulting bone direction explicitly: rest arm-down (0,-1,0)
    # must land on Roblox forward (0,0,-1).
    assert vec_approx(math3d.quat_rotate(q, (0, -1, 0)), (0, 0, -1), 1e-6)
    # Upper arm unaffected: still exactly the rest correction.
    corrections = r15.compute_rest_corrections(bvh_to_r15.conversion_quat("z"))
    assert quat_close(frames[0]["LeftUpperArm"][1], corrections["LeftUpperArm"], 1e-6)


def test_root_translation():
    """Move the hips +10cm up and +50cm forward (BVH +Z); the LowerTorso Pose
    (Root Motor6D) must translate up and toward Roblox -Z, scaled cm -> studs.
    The HumanoidRootPart Pose stays identity: Roblox's Animator ignores HRP
    Pose translation (verified in Studio), so root motion rides on LowerTorso."""
    anim = bvh_mod.Bvh.from_file(TPOSE_PATH)
    _set_channels(anim, 0, "Hips", {"Xposition": 0.0, "Yposition": 110.0, "Zposition": 50.0})
    frames, _ = bvh_to_r15.retarget_animation(anim, forward_axis="z", studs_per_meter=3.571)
    trans, q = frames[0]["LowerTorso"]
    s = 3.571 / 100.0
    assert vec_approx(trans, (0.0, 10.0 * s, -50.0 * s), 1e-6), trans
    assert quat_close(q, math3d.QUAT_IDENTITY, 1e-6)
    hrp_trans, hrp_q = frames[0]["HumanoidRootPart"]
    assert vec_approx(hrp_trans, (0.0, 0.0, 0.0), 1e-9), hrp_trans
    assert quat_close(hrp_q, math3d.QUAT_IDENTITY, 1e-9)

    # Root joint rotation must land on LowerTorso, not HumanoidRootPart.
    anim2 = bvh_mod.Bvh.from_file(TPOSE_PATH)
    _set_channels(anim2, 0, "Root", {"Yrotation": 90.0})
    frames2, _ = bvh_to_r15.retarget_animation(anim2, forward_axis="z")
    assert quat_close(frames2[0]["HumanoidRootPart"][1], math3d.QUAT_IDENTITY, 1e-6)
    assert approx(DEG(math3d.quat_angle(frames2[0]["LowerTorso"][1])), 90.0, 1e-3)


# ---------------------------------------------------------------------------
# Keyframe reduction and resampling
# ---------------------------------------------------------------------------


def _synthetic_frames(n, angle_fn):
    """Frames rotating LowerTorso about Y by angle_fn(i) degrees."""
    frames = []
    for i in range(n):
        f = {part: ((0.0, 0.0, 0.0), math3d.QUAT_IDENTITY) for part in r15.PART_ORDER}
        q = math3d.quat_from_axis_angle((0, 1, 0), math.radians(angle_fn(i)))
        f["LowerTorso"] = ((0.0, 0.0, 0.0), q)
        frames.append(f)
    return frames


def test_keyframe_reduction():
    # Constant animation: collapses to first + last frame.
    frames = _synthetic_frames(30, lambda i: 0.0)
    kept = bvh_to_r15.reduce_keyframes(frames, 0.5)
    assert kept == [0, 29], kept

    # Perfectly linear ramp: slerp reproduces it, so ends only.
    frames = _synthetic_frames(30, lambda i: 2.0 * i)
    kept = bvh_to_r15.reduce_keyframes(frames, 0.5)
    assert kept == [0, 29], kept

    # A corner in the middle must be kept.
    frames = _synthetic_frames(21, lambda i: 3.0 * min(i, 10))
    kept = bvh_to_r15.reduce_keyframes(frames, 0.5)
    assert 10 in kept, kept
    assert kept[0] == 0 and kept[-1] == 20

    # Epsilon 0 disables reduction.
    kept = bvh_to_r15.reduce_keyframes(frames, 0.0)
    assert kept == list(range(21))

    # Position changes are preserved too (translation epsilon).
    frames = _synthetic_frames(11, lambda i: 0.0)
    p0, q0 = frames[5]["HumanoidRootPart"]
    frames[5] = dict(frames[5])
    frames[5]["HumanoidRootPart"] = ((0.0, 1.0, 0.0), q0)
    kept = bvh_to_r15.reduce_keyframes(frames, 0.5)
    assert 5 in kept, kept


def test_resample():
    frames = _synthetic_frames(31, lambda i: 3.0 * i)  # 1s @ 30fps, 0..90 deg
    out, fps = bvh_to_r15.resample(frames, 30.0, 15.0)
    assert fps == 15.0
    assert len(out) == 16, len(out)  # 1s @ 15fps inclusive
    # Midpoint (t=0.5s) should be ~45 deg.
    ang = DEG(math3d.quat_angle(out[7]["LowerTorso"][1]))
    expected = 3.0 * (7 / 15.0) * 30.0
    assert approx(ang, expected, 1e-3), (ang, expected)
    # Endpoints preserved.
    assert quat_close(out[0]["LowerTorso"][1], frames[0]["LowerTorso"][1], 1e-6)
    assert quat_close(out[-1]["LowerTorso"][1], frames[-1]["LowerTorso"][1], 1e-6)

    # No-op when fps matches.
    same, fps2 = bvh_to_r15.resample(frames, 30.0, 30.0)
    assert same is frames and fps2 == 30.0


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def test_cli_smoke():
    out_dir = tempfile.mkdtemp(prefix="retarget_cli_")
    out_path = os.path.join(out_dir, "cli.rbxmx")
    rc = bvh_to_r15.main([
        TPOSE_PATH, "--output", out_path, "--loop", "--priority", "Movement",
        "--scale-studs-per-meter", "3.571", "--forward-axis", "z",
        "--keyframe-reduction-epsilon", "0.5", "--name", "cli_test",
    ])
    assert rc == 0
    tree = ET.parse(out_path)
    seq = tree.getroot().find("Item[@class='KeyframeSequence']")
    assert seq.find("Properties/string[@name='Name']").text == "cli_test"
    assert seq.find("Properties/token[@name='Priority']").text == "1"  # Movement


# ---------------------------------------------------------------------------


def main():
    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS %s" % name)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            import traceback
            print("FAIL %s: %s" % (name, exc))
            traceback.print_exc()
    print("%d/%d tests passed" % (len(tests) - failures, len(tests)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
