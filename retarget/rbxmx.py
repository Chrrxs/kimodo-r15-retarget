"""Writer for Roblox XML model files (.rbxmx) containing a KeyframeSequence.

Produced structure (validated against the Roblox XML model schema):

    <roblox version="4">
      <Item class="KeyframeSequence" referent="RBX0">
        <Properties>
          <string name="Name">...</string>
          <bool name="Loop">true|false</bool>
          <token name="Priority">N</token>          (Enum.AnimationPriority)
        </Properties>
        <Item class="Keyframe" referent="...">
          <Properties>
            <string name="Name">Keyframe</string>
            <float name="Time">t</float>
          </Properties>
          <Item class="Pose" referent="...">
            <Properties>
              <string name="Name">HumanoidRootPart</string>
              <CoordinateFrame name="CFrame">
                <X/><Y/><Z/><R00/>...<R22/>          (rotation row-major)
              </CoordinateFrame>
              <float name="Weight">1</float>
              <float name="MaskWeight">0</float>
              <token name="EasingDirection">0</token>
              <token name="EasingStyle">0</token>
            </Properties>
            ... nested child Pose items mirroring the rig hierarchy ...
          </Item>
        </Item>
        ...
      </Item>
    </roblox>

Notes:
- Pose's CFrame property is serialized as a ``CoordinateFrame`` element whose
  ``name`` attribute is ``CFrame`` (this is how Studio serializes Pose).
- Token values: Enum.AnimationPriority {Idle=0, Movement=1, Action=2,
  Action2=3, Action3=4, Action4=5, Core=1000}; Enum.PoseEasingStyle Linear=0;
  Enum.PoseEasingDirection In=0.
- Everything is emitted with xml.etree.ElementTree, so the output is
  well-formed by construction.
"""

import xml.etree.ElementTree as ET

try:
    from . import math3d, r15
except ImportError:
    import math3d
    import r15

#: Enum.AnimationPriority name -> token value.
ANIMATION_PRIORITY = {
    "Idle": 0,
    "Movement": 1,
    "Action": 2,
    "Action2": 3,
    "Action3": 4,
    "Action4": 5,
    "Core": 1000,
}

_CFRAME_FIELDS = ("X", "Y", "Z", "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22")


def _fmt(x):
    """Compact float formatting; keeps files small but precise."""
    if x == int(x) and abs(x) < 1e15:
        return str(int(x))
    return repr(round(float(x), 9))


class _RefCounter:
    def __init__(self):
        self.n = 0

    def next(self):
        ref = "RBX%d" % self.n
        self.n += 1
        return ref


def _prop(parent, tag, name, text):
    el = ET.SubElement(parent, tag, {"name": name})
    el.text = text
    return el


def _cframe_prop(parent, name, position, quat):
    """CoordinateFrame property: translation + row-major rotation matrix."""
    el = ET.SubElement(parent, "CoordinateFrame", {"name": name})
    m = math3d.quat_to_mat3(quat)
    values = (
        position[0], position[1], position[2],
        m[0][0], m[0][1], m[0][2],
        m[1][0], m[1][1], m[1][2],
        m[2][0], m[2][1], m[2][2],
    )
    for field, value in zip(_CFRAME_FIELDS, values):
        f = ET.SubElement(el, field)
        f.text = _fmt(value)
    return el


def _pose_item(parent_el, part_name, transforms, refs, easing_style=0, easing_direction=0):
    """Emit one Pose item (and recurse into child parts of the pose tree).

    Args:
        transforms: dict part -> (position, quaternion). Missing parts get
            identity transforms so the tree is always complete.
    """
    item = ET.SubElement(parent_el, "Item", {"class": "Pose", "referent": refs.next()})
    props = ET.SubElement(item, "Properties")
    _prop(props, "string", "Name", part_name)
    position, quat = transforms.get(part_name, ((0.0, 0.0, 0.0), math3d.QUAT_IDENTITY))
    _cframe_prop(props, "CFrame", position, quat)
    _prop(props, "float", "Weight", "1")
    _prop(props, "float", "MaskWeight", "0")
    _prop(props, "token", "EasingDirection", str(easing_direction))
    _prop(props, "token", "EasingStyle", str(easing_style))
    for child in r15.PART_ORDER:
        if r15.POSE_PARENT[child] == part_name:
            _pose_item(item, child, transforms, refs, easing_style, easing_direction)
    return item


def build_keyframe_sequence(name, keyframes, loop=False, priority="Action"):
    """Build the <roblox> ElementTree for a KeyframeSequence.

    Args:
        name: animation name (KeyframeSequence.Name).
        keyframes: list of (time_seconds, transforms) where transforms maps
            part name -> ((x, y, z) studs, (w, x, y, z) quaternion).
        loop: KeyframeSequence.Loop.
        priority: Enum.AnimationPriority name (e.g. "Action") or int value.

    Returns:
        xml.etree.ElementTree.ElementTree rooted at <roblox>.
    """
    if isinstance(priority, str):
        if priority not in ANIMATION_PRIORITY:
            raise ValueError(
                "unknown AnimationPriority %r (expected one of %s)"
                % (priority, ", ".join(ANIMATION_PRIORITY))
            )
        priority = ANIMATION_PRIORITY[priority]
    if not keyframes:
        raise ValueError("keyframes must be non-empty")

    refs = _RefCounter()
    root = ET.Element("roblox", {"version": "4"})
    seq = ET.SubElement(root, "Item", {"class": "KeyframeSequence", "referent": refs.next()})
    seq_props = ET.SubElement(seq, "Properties")
    _prop(seq_props, "string", "Name", name)
    _prop(seq_props, "bool", "Loop", "true" if loop else "false")
    _prop(seq_props, "token", "Priority", str(priority))

    for time_s, transforms in keyframes:
        kf = ET.SubElement(seq, "Item", {"class": "Keyframe", "referent": refs.next()})
        kf_props = ET.SubElement(kf, "Properties")
        _prop(kf_props, "string", "Name", "Keyframe")
        _prop(kf_props, "float", "Time", _fmt(float(time_s)))
        _pose_item(kf, "HumanoidRootPart", transforms, refs)

    return ET.ElementTree(root)


def write_rbxmx(path, name, keyframes, loop=False, priority="Action"):
    """Build and write the KeyframeSequence .rbxmx file (UTF-8, LF, indented)."""
    tree = build_keyframe_sequence(name, keyframes, loop=loop, priority=priority)
    ET.indent(tree, space="  ")
    with open(path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=False)
        f.write(b"\n")
