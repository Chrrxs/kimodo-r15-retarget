"""Dependency-free BVH parser (stdlib only).

Parses the HIERARCHY and MOTION sections of a BVH file into a joint tree plus
a flat per-frame channel array.  Written for the kimodo/SOMA ``somaskel77``
exports but general enough for any conforming BVH.

Conventions handled:
- Any per-joint channel set/order (kimodo uses ``Zrotation Yrotation
  Xrotation`` everywhere; root joints additionally have
  ``Xposition Yposition Zposition``).
- ``End Site`` blocks are parsed and ignored (kimodo strips them anyway).
- Position channels, where present, are treated as the FULL local translation
  of the joint (they replace the static OFFSET rather than adding to it).
  This matches the kimodo writer, which emits Hips position channels equal to
  the Hips offset in the rest pose.
"""

try:
    from . import math3d
except ImportError:  # running as a plain script with retarget/ on sys.path
    import math3d

_ROT_CHANNELS = {"Xrotation": "X", "Yrotation": "Y", "Zrotation": "Z"}
_POS_CHANNELS = {"Xposition": 0, "Yposition": 1, "Zposition": 2}


class BvhJoint:
    """One joint in the BVH hierarchy."""

    __slots__ = ("name", "offset", "channels", "children", "parent", "channel_start")

    def __init__(self, name, parent=None):
        self.name = name
        self.offset = (0.0, 0.0, 0.0)
        self.channels = []           # channel names in file order
        self.children = []
        self.parent = parent
        self.channel_start = 0       # index of this joint's first channel in a frame row

    @property
    def rotation_order(self):
        """Axis letters of rotation channels in file order, e.g. 'ZYX'."""
        return "".join(_ROT_CHANNELS[c] for c in self.channels if c in _ROT_CHANNELS)

    @property
    def has_position(self):
        return any(c in _POS_CHANNELS for c in self.channels)

    def __repr__(self):
        return "BvhJoint(%r)" % self.name


class Bvh:
    """Parsed BVH file: joint tree + motion frames.

    Attributes:
        root: the ROOT BvhJoint.
        joints: all joints in file (depth-first) order, End Sites excluded.
        joint_map: name -> BvhJoint.
        frames: list of per-frame channel value lists (floats, degrees/units
            as in the file).
        frame_time: seconds per frame.
    """

    def __init__(self, text):
        self.root = None
        self.joints = []
        self.joint_map = {}
        self.frames = []
        self.frame_time = 1.0 / 30.0
        self._parse(text)

    @classmethod
    def from_file(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            return cls(f.read())

    @property
    def nframes(self):
        return len(self.frames)

    @property
    def fps(self):
        return 1.0 / self.frame_time

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse(self, text):
        tokens = text.split()
        i = 0
        n = len(tokens)

        def expect(tok):
            nonlocal i
            if i >= n or tokens[i] != tok:
                got = tokens[i] if i < n else "<eof>"
                raise ValueError("BVH parse error: expected %r, got %r (token %d)" % (tok, got, i))
            i += 1

        expect("HIERARCHY")
        if i >= n or tokens[i] != "ROOT":
            raise ValueError("BVH parse error: expected ROOT after HIERARCHY")

        channel_cursor = 0

        def parse_joint(parent):
            nonlocal i, channel_cursor
            kind = tokens[i]
            i += 1
            if kind == "End":  # "End Site" block: consume and ignore
                expect("Site")
                expect("{")
                expect("OFFSET")
                i += 3
                expect("}")
                return None
            name = tokens[i]
            i += 1
            joint = BvhJoint(name, parent)
            expect("{")
            expect("OFFSET")
            joint.offset = (float(tokens[i]), float(tokens[i + 1]), float(tokens[i + 2]))
            i += 3
            if tokens[i] == "CHANNELS":
                i += 1
                count = int(tokens[i])
                i += 1
                joint.channels = tokens[i : i + count]
                i += count
                joint.channel_start = channel_cursor
                channel_cursor += count
            self.joints.append(joint)
            if name in self.joint_map:
                raise ValueError("duplicate joint name %r in BVH" % name)
            self.joint_map[name] = joint
            while tokens[i] in ("JOINT", "End"):
                child = parse_joint(joint)
                if child is not None:
                    joint.children.append(child)
            expect("}")
            return joint

        self.root = parse_joint(None)
        self._total_channels = channel_cursor

        expect("MOTION")
        expect("Frames:")
        frame_count = int(tokens[i])
        i += 1
        expect("Frame")
        if tokens[i] != "Time:":
            raise ValueError("BVH parse error: expected 'Time:' after 'Frame'")
        i += 1
        self.frame_time = float(tokens[i])
        i += 1

        values = tokens[i:]
        expected = frame_count * channel_cursor
        if len(values) < expected:
            raise ValueError(
                "BVH motion data truncated: expected %d values (%d frames x %d channels), got %d"
                % (expected, frame_count, channel_cursor, len(values))
            )
        flat = [float(v) for v in values[:expected]]
        self.frames = [
            flat[f * channel_cursor : (f + 1) * channel_cursor] for f in range(frame_count)
        ]

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------

    def joint_rotation_quat(self, joint_name, frame_index):
        """Local rotation of a joint at a frame as a (w,x,y,z) quaternion.

        Composes the rotation channels intrinsically in file order (kimodo:
        ZYX, i.e. R = Rz @ Ry @ Rx).
        """
        joint = self.joint_map[joint_name]
        row = self.frames[frame_index]
        degrees = []
        order = ""
        for k, ch in enumerate(joint.channels):
            if ch in _ROT_CHANNELS:
                order += _ROT_CHANNELS[ch]
                degrees.append(row[joint.channel_start + k])
        if not order:
            return math3d.QUAT_IDENTITY
        return math3d.quat_from_bvh_euler(order, degrees)

    def joint_translation(self, joint_name, frame_index):
        """Local translation of a joint at a frame (file units, i.e. cm).

        If the joint has position channels their values are the local
        translation (they replace the static offset -- kimodo convention);
        otherwise the static OFFSET is returned.
        """
        joint = self.joint_map[joint_name]
        if not joint.has_position:
            return joint.offset
        row = self.frames[frame_index]
        pos = [0.0, 0.0, 0.0]
        for k, ch in enumerate(joint.channels):
            if ch in _POS_CHANNELS:
                pos[_POS_CHANNELS[ch]] = row[joint.channel_start + k]
        return tuple(pos)

    def rest_world_position(self, joint_name):
        """Rest-pose world position of a joint: sum of OFFSETs up the chain."""
        joint = self.joint_map[joint_name]
        pos = (0.0, 0.0, 0.0)
        while joint is not None:
            pos = math3d.vec_add(pos, joint.offset)
            joint = joint.parent
        return pos
