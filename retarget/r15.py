"""Roblox R15 rig description and the SOMA(somaskel77) -> R15 joint mapping.

Rest-pose correction convention
-------------------------------
Both skeletons have "identity" rest poses in the sense that every joint frame
has zero rotation at rest (the SOMA standard T-pose BVH stores all-zero
rotations; an R15 rig from the Rig Builder has all Motor6D Transforms at
identity).  The two rest poses are nevertheless PHYSICALLY different: the SOMA
T-pose holds the arms horizontal (left arm along +X) while the R15 rest pose
has the arms hanging straight down (-Y).

For every mapped part we therefore declare two unit "bone direction" vectors:

- ``src_dir``: the bone direction in the SOMA rest pose, in BVH axes.
- ``tgt_dir``: the bone direction in the R15 rest pose, in Roblox axes.

At runtime, with C the BVH->Roblox axis-conversion rotation, we compute the
per-part rest correction as the minimal rotation

    D_part = quat_between(tgt_dir, C(src_dir))     # D * tgt_dir == C(src_dir)

and retarget local rotations with

    T_part(t) = D_parent^-1 * (C * L_src(t) * C^-1) * D_part

where ``L_src`` is the composed local rotation of the mapped SOMA chain and
``parent`` is the parent part in the R15 pose tree.  At rest (L_src = I) this
yields T = D_parent^-1 * D_part: identity everywhere except the shoulder
joints, which get the expected +/-90 degree roll that carries the T-pose arms
down into the R15 rest pose.  Because elbow/wrist (and knee/ankle) share their
parent's bone-direction mismatch, their corrections cancel to identity.

The minimal rotation leaves the twist about the bone axis unconstrained; this
is a canonical choice that may need per-joint tuning after visual validation.
"""

try:
    from . import math3d
except ImportError:
    import math3d

# ---------------------------------------------------------------------------
# R15 pose tree: part -> parent part (mirrors the Motor6D/part hierarchy).
# The Keyframe's Pose items must nest exactly like this.
# ---------------------------------------------------------------------------

POSE_PARENT = {
    "HumanoidRootPart": None,
    "LowerTorso": "HumanoidRootPart",
    "UpperTorso": "LowerTorso",
    "Head": "UpperTorso",
    "LeftUpperArm": "UpperTorso",
    "LeftLowerArm": "LeftUpperArm",
    "LeftHand": "LeftLowerArm",
    "RightUpperArm": "UpperTorso",
    "RightLowerArm": "RightUpperArm",
    "RightHand": "RightLowerArm",
    "LeftUpperLeg": "LowerTorso",
    "LeftLowerLeg": "LeftUpperLeg",
    "LeftFoot": "LeftLowerLeg",
    "RightUpperLeg": "LowerTorso",
    "RightLowerLeg": "RightUpperLeg",
    "RightFoot": "RightLowerLeg",
}

#: Parts in parent-before-child order (safe iteration order for FK-style work).
PART_ORDER = list(POSE_PARENT.keys())

#: Motor6D that animates each part (Part1 side).  Informational: the .rbxmx
#: Pose items are named after PARTS, but Studio maps them onto these motors.
MOTOR6D_OF_PART = {
    "LowerTorso": "Root",
    "UpperTorso": "Waist",
    "Head": "Neck",
    "LeftUpperArm": "LeftShoulder",
    "LeftLowerArm": "LeftElbow",
    "LeftHand": "LeftWrist",
    "RightUpperArm": "RightShoulder",
    "RightLowerArm": "RightElbow",
    "RightHand": "RightWrist",
    "LeftUpperLeg": "LeftHip",
    "LeftLowerLeg": "LeftKnee",
    "LeftFoot": "LeftAnkle",
    "RightUpperLeg": "RightHip",
    "RightLowerLeg": "RightKnee",
    "RightFoot": "RightAnkle",
}

# ---------------------------------------------------------------------------
# SOMA -> R15 mapping.
#
# chain:    SOMA joints whose LOCAL rotations are composed (in order, i.e.
#           q = q_first * ... * q_last) to drive the part's Motor6D.
# src_dir:  rest bone direction in BVH axes (SOMA standard T-pose).
#           Declared explicitly (see module docstring); the values match the
#           somaskel77 T-pose offsets to within a few degrees and the test
#           suite cross-checks them against the reference BVH.
# tgt_dir:  rest bone direction in Roblox axes (R15 Rig Builder rest pose:
#           torso/head up, arms straight down, legs down, feet forward).
#           Roblox characters face -Z, so "foot forward" is (0, 0, -1).
# ---------------------------------------------------------------------------

_UP = (0.0, 1.0, 0.0)
_DOWN = (0.0, -1.0, 0.0)

# SOMA's standard-T-pose neck is NOT vertical: the Neck1->Head segment leans
# ~17 deg forward (offsets sum to (0, 13.84, 4.26) cm in the somaskel77
# reference BVH). Motion data expresses "head level" as a ~+18 deg pitch-up
# of the neck chain relative to that leaned rest, so declaring the neck as
# vertical made retargeted heads tilt ~20 deg upward (observed in Studio).
# Using the true rest direction folds the lean into the rest correction.
_NECK_DIR = (0.0, 0.9557, 0.2943)  # normalize((0, 13.84, 4.26))

MAPPING = {
    # BVH "Root" wrapper + Hips compose onto LowerTorso (Root Motor6D).
    "LowerTorso": {"chain": ["Root", "Hips"], "src_dir": _UP, "tgt_dir": _UP},
    "UpperTorso": {"chain": ["Spine1", "Spine2", "Chest"], "src_dir": _UP, "tgt_dir": _UP},
    "Head": {"chain": ["Neck1", "Neck2", "Head"], "src_dir": _NECK_DIR, "tgt_dir": _UP},
    "LeftUpperArm": {"chain": ["LeftShoulder", "LeftArm"], "src_dir": (1.0, 0.0, 0.0), "tgt_dir": _DOWN},
    "LeftLowerArm": {"chain": ["LeftForeArm"], "src_dir": (1.0, 0.0, 0.0), "tgt_dir": _DOWN},
    "LeftHand": {"chain": ["LeftHand"], "src_dir": (1.0, 0.0, 0.0), "tgt_dir": _DOWN},
    "RightUpperArm": {"chain": ["RightShoulder", "RightArm"], "src_dir": (-1.0, 0.0, 0.0), "tgt_dir": _DOWN},
    "RightLowerArm": {"chain": ["RightForeArm"], "src_dir": (-1.0, 0.0, 0.0), "tgt_dir": _DOWN},
    "RightHand": {"chain": ["RightHand"], "src_dir": (-1.0, 0.0, 0.0), "tgt_dir": _DOWN},
    "LeftUpperLeg": {"chain": ["LeftLeg"], "src_dir": _DOWN, "tgt_dir": _DOWN},
    "LeftLowerLeg": {"chain": ["LeftShin"], "src_dir": _DOWN, "tgt_dir": _DOWN},
    # Feet: forward in BVH is +Z (SOMA faces +Z), forward in Roblox is -Z.
    "LeftFoot": {"chain": ["LeftFoot"], "src_dir": (0.0, 0.0, 1.0), "tgt_dir": (0.0, 0.0, -1.0)},
    "RightUpperLeg": {"chain": ["RightLeg"], "src_dir": _DOWN, "tgt_dir": _DOWN},
    "RightLowerLeg": {"chain": ["RightShin"], "src_dir": _DOWN, "tgt_dir": _DOWN},
    "RightFoot": {"chain": ["RightFoot"], "src_dir": (0.0, 0.0, 1.0), "tgt_dir": (0.0, 0.0, -1.0)},
}

#: SOMA joints the retargeter needs to find in the input BVH.
REQUIRED_SOMA_JOINTS = sorted({j for m in MAPPING.values() for j in m["chain"]})


def compute_rest_corrections(conv_quat):
    """Per-part rest-correction quaternions D_part.

    Args:
        conv_quat: axis-conversion rotation C (BVH -> Roblox) as a quaternion.

    Returns:
        dict part_name -> quaternion D with D * tgt_dir == C(src_dir).
        HumanoidRootPart (never rotated) gets the identity.
    """
    corrections = {"HumanoidRootPart": math3d.QUAT_IDENTITY}
    for part, spec in MAPPING.items():
        src_in_rbx = math3d.quat_rotate(conv_quat, spec["src_dir"])
        corrections[part] = math3d.quat_between(spec["tgt_dir"], src_in_rbx)
    return corrections
