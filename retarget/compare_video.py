#!/usr/bin/env python3
"""Side-by-side comparison video: kimodo source motion vs retargeted R15.

Left panel : SOMA skeleton, world positions via the same BVH parser/FK the
             retargeter uses (amber = bones the retarget consumes).
Right panel: R15 rig reconstructed by forward kinematics from the retargeted
             Motor6D transforms using real rig geometry captured from Studio
             (r15_rig_geometry.json) — i.e. exactly what Roblox's Animator
             computes: part1.CFrame = part0.CFrame * A0 * Transform * A1^-1.

Both panels share one camera (the R15 side is drawn in cm via the same
studs-per-meter scale the retargeter used, and its yaw is offset 180 deg so
SOMA +Z forward and Roblox -Z forward read as the same on-screen direction).

Usage:
  python3 retarget/compare_video.py <input.bvh> [--output out.mp4]
      [--forward-axis z] [--scale-studs-per-meter 3.571] [--size 1280x720]

Requires pillow; encodes with the imageio-ffmpeg bundled ffmpeg (or `ffmpeg`
on PATH).
"""
import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bvh as bvh_mod  # noqa: E402
import bvh_to_r15  # noqa: E402
import math3d  # noqa: E402
import r15  # noqa: E402

GEOMETRY_PATH = Path(__file__).resolve().parent / "r15_rig_geometry.json"

# Palette (matches the motion-viewer artifact).
BG = (20, 23, 28)
GRID = (42, 47, 55)
TEXT = (201, 206, 214)
TEXT_DIM = (123, 130, 142)
ACCENT = (232, 163, 61)      # mapped SOMA bones
IGNORED = (74, 80, 90)       # unmapped SOMA bones
R15_BODY = (91, 155, 213)    # R15 part edges
R15_HEAD = (168, 200, 235)
DIVIDER = (42, 47, 55)


# --- minimal CFrame ops as (pos, quat) --------------------------------------
def cf_mul(a, b):
    pa, qa = a
    pb, qb = b
    return (math3d.vec_add(pa, math3d.quat_rotate(qa, pb)), math3d.quat_mul(qa, qb))


def cf_translation(v):
    return (tuple(v), math3d.QUAT_IDENTITY)


CF_IDENTITY = ((0.0, 0.0, 0.0), math3d.QUAT_IDENTITY)


def fk_r15(transforms, geometry, root_cf):
    """Part CFrames from Motor6D transforms (the Animator's joint formula)."""
    cfs = {"HumanoidRootPart": root_cf}
    joints = list(geometry["joints"])
    # Joints are listed parents-first, one pass suffices; loop defensively.
    for _ in range(4):
        for j in joints:
            if j["part0"] in cfs and j["part1"] not in cfs:
                t = transforms.get(j["part1"], ((0.0, 0.0, 0.0), math3d.QUAT_IDENTITY))
                cf = cf_mul(cfs[j["part0"]], cf_translation(j["a0"]))
                cf = cf_mul(cf, t)
                cf = cf_mul(cf, cf_translation((-j["a1"][0], -j["a1"][1], -j["a1"][2])))
                cfs[j["part1"]] = cf
    return cfs


def rest_floor_offset(geometry):
    """Y offset that puts the rest-pose feet bottoms on y=0 (HRP at origin)."""
    cfs = fk_r15({}, geometry, CF_IDENTITY)
    lowest = min(
        cfs[p][0][1] - geometry["parts"][p][1] / 2.0 for p in ("LeftFoot", "RightFoot")
    )
    return -lowest


class Camera:
    """Orbit camera, same projection as the HTML viewer (cm units, y-up)."""

    def __init__(self, yaw, pitch, dist, width, height):
        self.yaw, self.pitch, self.dist = yaw, pitch, dist
        self.w, self.h = width, height
        self.target = [0.0, 90.0, 0.0]

    def project(self, p):
        x = p[0] - self.target[0]
        y = p[1] - self.target[1]
        z = p[2] - self.target[2]
        cy, sy = math.cos(self.yaw), math.sin(self.yaw)
        x1, z1 = cy * x + sy * z, -sy * x + cy * z
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        y2, z2 = cp * y - sp * z1, sp * y + cp * z1
        z2 += self.dist
        if z2 < 10:
            return None
        f = 1.15 * min(self.w, self.h)
        return (self.w / 2 + f * x1 / z2, self.h / 2 - f * y2 / z2, z2)


def draw_grid(dr, cam, ox=0):
    g, n = 50, 10
    cx = round(cam.target[0] / g) * g
    cz = round(cam.target[2] / g) * g
    for i in range(-n, n + 1):
        for a, b in (
            ((cx + i * g, 0, cz - n * g), (cx + i * g, 0, cz + n * g)),
            ((cx - n * g, 0, cz + i * g), (cx + n * g, 0, cz + i * g)),
        ):
            pa, pb = cam.project(a), cam.project(b)
            if pa and pb:
                dr.line([(pa[0] + ox, pa[1]), (pb[0] + ox, pb[1])], fill=GRID, width=1)


def world_positions(anim, frame):
    pos, rot, out = {}, {}, {}
    for joint in anim.joints:
        q = anim.joint_rotation_quat(joint.name, frame)
        t = anim.joint_translation(joint.name, frame)
        if joint.parent is None:
            pos[joint.name], rot[joint.name] = t, q
        else:
            pq = rot[joint.parent.name]
            pos[joint.name] = math3d.vec_add(pos[joint.parent.name], math3d.quat_rotate(pq, t))
            rot[joint.name] = math3d.quat_mul(pq, q)
        out[joint.name] = pos[joint.name]
    return out


BOX_EDGES = [
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]


def draw_r15(dr, cam, cfs, geometry, cm_per_stud, ox):
    """Wireframe boxes, painter-sorted far-to-near."""
    items = []
    for part, (size_x, size_y, size_z) in geometry["parts"].items():
        if part == "HumanoidRootPart":
            continue
        pos, q = cfs[part]
        hx, hy, hz = size_x / 2, size_y / 2, size_z / 2
        corners = []
        depth = 0.0
        for i in range(8):
            local = (hx if i & 1 else -hx, hy if i & 2 else -hy, hz if i & 4 else -hz)
            wp = math3d.vec_add(pos, math3d.quat_rotate(q, local))
            sp = cam.project((wp[0] * cm_per_stud, wp[1] * cm_per_stud, wp[2] * cm_per_stud))
            if sp is None:
                corners = None
                break
            corners.append(sp)
            depth += sp[2]
        if corners:
            items.append((depth / 8, part, corners))
    items.sort(reverse=True)
    for _, part, c in items:
        color = R15_HEAD if part == "Head" else R15_BODY
        for a, b in BOX_EDGES:
            dr.line([(c[a][0] + ox, c[a][1]), (c[b][0] + ox, c[b][1])], fill=color, width=2)


def draw_soma(dr, cam, anim, positions, ox):
    mapped = set(r15.REQUIRED_SOMA_JOINTS)
    for pass_mapped in (False, True):
        for joint in anim.joints:
            if joint.parent is None:
                continue
            # Root is a stationary wrapper joint (trajectory lives on Hips);
            # the Root->Hips segment is not a bone and would draw a giant
            # line back to the world origin.
            if joint.parent.parent is None:
                continue
            is_mapped = joint.name in mapped and joint.parent.name in mapped
            if is_mapped != pass_mapped:
                continue
            a = cam.project(positions[joint.name])
            b = cam.project(positions[joint.parent.name])
            if a and b:
                dr.line(
                    [(a[0] + ox, a[1]), (b[0] + ox, b[1])],
                    fill=ACCENT if is_mapped else IGNORED,
                    width=3 if is_mapped else 1,
                )


def find_ffmpeg():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        exe = shutil.which("ffmpeg")
        if exe:
            return exe
        sys.exit("ffmpeg not found: pip install imageio-ffmpeg, or install ffmpeg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--output", default=None)
    ap.add_argument("--forward-axis", default="z", choices=["z", "-z", "x", "-x"])
    ap.add_argument("--scale-studs-per-meter", type=float, default=3.571)
    ap.add_argument("--size", default="1280x720")
    args = ap.parse_args()

    width, height = (int(v) for v in args.size.split("x"))
    panel_w = width // 2
    out_path = args.output or str(Path(args.input).with_suffix("")) + "_compare.mp4"

    anim = bvh_mod.Bvh.from_file(args.input)
    frames, fps = bvh_to_r15.retarget_animation(
        anim, forward_axis=args.forward_axis, studs_per_meter=args.scale_studs_per_meter
    )
    geometry = json.loads(GEOMETRY_PATH.read_text())
    cm_per_stud = 100.0 / args.scale_studs_per_meter
    hrp_y = rest_floor_offset(geometry)

    # Both panels use the SAME camera view. The R15 world is rotated 180 deg
    # about Y before rendering (proper rotation, no mirroring) so the R15
    # character faces +Z like SOMA and the two views are visually identical.
    cam_l = Camera(-0.55, 0.28, 430, panel_w, height - 40)
    cam_r = Camera(-0.55, 0.28, 430, panel_w, height - 40)
    flip_q = math3d.quat_from_axis_angle((0.0, 1.0, 0.0), math.pi)

    ffmpeg = find_ffmpeg()
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{fps:.3f}", "-i", "-",
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", out_path],
        stdin=subprocess.PIPE,
    )

    n = anim.nframes
    for f in range(n):
        img = Image.new("RGB", (width, height), BG)

        # Each panel renders into its own image so nothing bleeds across.
        panel_l = Image.new("RGB", (panel_w, height - 40), BG)
        dr_l = ImageDraw.Draw(panel_l)
        positions = world_positions(anim, f)
        hips = positions["Hips"]
        cam_l.target[0] += (hips[0] - cam_l.target[0]) * 0.15
        cam_l.target[2] += (hips[2] - cam_l.target[2]) * 0.15
        draw_grid(dr_l, cam_l, 0)
        draw_soma(dr_l, cam_l, anim, positions, 0)
        img.paste(panel_l, (0, 0))

        panel_r = Image.new("RGB", (panel_w, height - 40), BG)
        dr_r = ImageDraw.Draw(panel_r)
        cfs = fk_r15(frames[f], geometry, ((0.0, hrp_y, 0.0), math3d.QUAT_IDENTITY))
        cfs = {
            part: (math3d.quat_rotate(flip_q, pos), math3d.quat_mul(flip_q, q))
            for part, (pos, q) in cfs.items()
        }
        lt = cfs["LowerTorso"][0]
        cam_r.target[0] += (lt[0] * cm_per_stud - cam_r.target[0]) * 0.15
        cam_r.target[2] += (lt[2] * cm_per_stud - cam_r.target[2]) * 0.15
        draw_grid(dr_r, cam_r, 0)
        draw_r15(dr_r, cam_r, cfs, geometry, cm_per_stud, 0)
        img.paste(panel_r, (panel_w, 0))

        # Chrome: divider, labels, timecode.
        dr = ImageDraw.Draw(img)
        dr.rectangle([panel_w - 1, 0, panel_w + 1, height], fill=DIVIDER)
        dr.rectangle([0, height - 40, width, height], fill=(28, 32, 39))
        dr.text((12, height - 30), "KIMODO SOURCE (somaskel77 BVH)", fill=ACCENT)
        dr.text((panel_w + 12, height - 30), "ROBLOX R15 (retargeted Motor6D FK)", fill=R15_BODY)
        tc = f"frame {f:>4} / {n - 1}   t = {f / fps:5.2f}s"
        dr.text((width - 12 - dr.textlength(tc), height - 30), tc, fill=TEXT_DIM)

        proc.stdin.write(img.tobytes())

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        sys.exit(f"ffmpeg failed with exit code {proc.returncode}")
    print(f"wrote {out_path}: {n} frames @ {fps:.1f} fps, {width}x{height}")


if __name__ == "__main__":
    main()
