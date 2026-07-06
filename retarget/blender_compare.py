"""Blender-rendered side-by-side retarget comparison.

Renders ONE scene with two solid characters animating simultaneously:
  LEFT : a mannequin (capsules per bone) driven directly by the kimodo BVH.
  RIGHT: a blocky Roblox R15 character driven by precomputed world CFrames
         from r15_fk.json (exported by retarget/export_fk_json.py).

Run headless with Blender 4.2 LTS (WORKBENCH engine, CPU only):

    blender -b --factory-startup -noaudio \
        --python retarget/blender_compare.py -- \
        --bvh path/to/motion.bvh \
        --fk  path/to/r15_fk.json \
        --output-dir /tmp/frames [--size 1280x720] [--stills 1,75,150]

End-to-end (FK export if missing + render + h264 encode) is wrapped by
retarget/blender_compare_run.sh, or `make compare-blender RETARGET_INPUT=...`.
Frames are written as PNGs into --output-dir; encode them with e.g.:

    ffmpeg -framerate 30 -i frame_%04d.png -c:v libx264 -pix_fmt yuv420p \
        -crf 20 motion_compare_blender.mp4
"""

import argparse
import json
import math
import os
import sys

import bpy
from mathutils import Matrix, Quaternion, Vector

# ---------------------------------------------------------------- constants

# Project QA palette.
AMBER = (0.91, 0.64, 0.24, 1.0)          # mannequin (kimodo source)
R15_BLUE = (0.36, 0.61, 0.84, 1.0)       # accent (labels)
# Real rig colour: RGB (163,162,165)/255 sRGB -> linear for diffuse_color.
R15_GREY = (0.372, 0.366, 0.381, 1.0)
GROUND_GREY = (0.115, 0.12, 0.13, 1.0)
BACKGROUND = (0.022, 0.027, 0.038)       # ~#14171c after display transform

R15_X_OFFSET = 2.2      # metres, applied after Roblox->Blender axis conversion
CAM_DIST = 7.2
CAM_ANGLE = math.radians(35.0)  # off the travel axis (-Y)
CAM_HEIGHT = 1.6
TRACK_SMOOTH = 13       # moving-average window (frames) for the camera target

# Mannequin bone spec: name -> capsule radius (m). Fingers/eyes/jaw skipped.
BONE_RADII = {
    "Hips": 0.095,
    "Spine1": 0.100,
    "Spine2": 0.105,
    "Chest": 0.105,
    "Neck1": 0.038,
    "Neck2": 0.038,
    "LeftShoulder": 0.042, "RightShoulder": 0.042,
    "LeftArm": 0.042, "RightArm": 0.042,
    "LeftForeArm": 0.036, "RightForeArm": 0.036,
    "LeftHand": 0.030, "RightHand": 0.030,
    "LeftLeg": 0.058, "RightLeg": 0.058,
    "LeftShin": 0.048, "RightShin": 0.048,
    "LeftFoot": 0.038, "RightFoot": 0.038,
    "LeftToeBase": 0.032, "RightToeBase": 0.032,
}
HEAD_BONE = "Head"
HEAD_RADIUS = 0.11

STUDS_PER_METER_DEFAULT = 3.571

# Real rig meshes (part-local studs, y-up, 1:1 with the part) live next to
# this script; the classic built-in Head has no OBJ and is approximated as a
# rounded cylinder (see r15_rig_geometry_kimodo.json head_special_mesh note).
RIG_MESH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "kimodo_rig")
HEAD_CYL_DIAMETER = 1.2   # studs
HEAD_CYL_HEIGHT = 1.25    # studs

# Roblox (y-up, RH) -> Blender (z-up): +90 degrees about X.
Q_CONV = Quaternion((1.0, 0.0, 0.0), math.radians(90.0))


def rbx_pos_to_blender(p, spm):
    x, y, z = (c / spm for c in p)
    return Vector((x, -z, y))


# ------------------------------------------------------------------ helpers

def fail(msg):
    print("ERROR: %s" % msg, file=sys.stderr)
    sys.exit(1)


def parse_args():
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bvh", required=True)
    ap.add_argument("--fk", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--stills", default="",
                    help="comma-separated frame numbers; render only these")
    return ap.parse_args(argv)


def make_material(name, rgba):
    mat = bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    mat.roughness = 0.6
    return mat


def link(obj):
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_from_pydata(name, verts, faces):
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    me.validate()
    me.update()
    return me


def make_capsule_mesh(name, radius, length, segments=16, rings=6):
    """Capsule along -Y from y=0 (bone tail) to y=-length (bone head)."""
    length = max(length, 1e-4)
    cyl = max(length - 2.0 * radius, 0.0)
    # Build as a UV-sphere split at the equator with a cylindrical midsection,
    # axis along Y, top hemisphere centred at y=-radius_top_off.
    top_c = -min(radius, length * 0.5)
    bot_c = -length + min(radius, length * 0.5)
    if bot_c > top_c:
        top_c = bot_c = -length * 0.5
    r = radius
    verts, faces = [], []
    # latitude rows: top pole .. equator (top hemi), equator (bottom hemi) .. pole
    rows = []
    for i in range(rings + 1):            # top hemisphere: 90 -> 0 deg
        phi = math.pi / 2 * (1 - i / rings)
        rows.append((r * math.cos(phi), top_c + r * math.sin(phi)))
    for i in range(1, rings + 1):         # bottom hemisphere: 0 -> -90 deg
        phi = -math.pi / 2 * (i / rings)
        rows.append((r * math.cos(phi), bot_c + r * math.sin(phi)))
    for rad, y in rows:
        for s in range(segments):
            a = 2 * math.pi * s / segments
            verts.append((rad * math.cos(a), y, rad * math.sin(a)))
    nrows = len(rows)
    for i in range(nrows - 1):
        for s in range(segments):
            a = i * segments + s
            b = i * segments + (s + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    me = mesh_from_pydata(name, verts, faces)
    for p in me.polygons:
        p.use_smooth = True
    return me


def make_uv_sphere_mesh(name, radius, segments=20, rings=12):
    verts, faces = [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        y = radius * math.cos(phi)
        rad = radius * math.sin(phi)
        for s in range(segments):
            a = 2 * math.pi * s / segments
            verts.append((rad * math.cos(a), y, rad * math.sin(a)))
    for i in range(rings):
        for s in range(segments):
            a = i * segments + s
            b = i * segments + (s + 1) % segments
            faces.append((a, b, b + segments, a + segments))
    me = mesh_from_pydata(name, verts, faces)
    for p in me.polygons:
        p.use_smooth = True
    return me


def make_cylinder_mesh(name, radius, height, segments=24):
    """Cylinder with axis along local Y (roblox part 'up'), centred."""
    verts, faces = [], []
    hy = height / 2
    for y in (-hy, hy):
        for s in range(segments):
            a = 2 * math.pi * s / segments
            verts.append((radius * math.cos(a), y, radius * math.sin(a)))
    for s in range(segments):
        s2 = (s + 1) % segments
        faces.append((s, s2, s2 + segments, s + segments))
    faces.append(tuple(range(segments - 1, -1, -1)))
    faces.append(tuple(range(segments, 2 * segments)))
    me = mesh_from_pydata(name, verts, faces)
    for p in me.polygons:
        p.use_smooth = len(p.vertices) == 4
    return me


def import_part_obj(part, spm):
    """Import kimodo_rig/<part>.obj keeping raw file coordinates.

    Vertices are part-local studs, roblox axes. The keyframed object rotation
    (Q_CONV @ q_roblox) already maps roblox-local into Blender world, so the
    mesh itself must NOT be axis-converted -- only scaled studs->metres.
    forward=-Y/up=Z below is the identity mapping for the importer.
    """
    path = os.path.join(RIG_MESH_DIR, part + ".obj")
    if not os.path.isfile(path):
        fail("rig mesh not found: %s" % path)
    bpy.ops.wm.obj_import(filepath=path, forward_axis="NEGATIVE_Y",
                          up_axis="Z")
    obj = bpy.context.selected_objects[0]
    obj.name = "r15_" + part
    obj.data.transform(Matrix.Scale(1.0 / spm, 4))
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)
    obj.data.materials.clear()
    return obj


def make_box_mesh(name, sx, sy, sz):
    hx, hy, hz = sx / 2, sy / 2, sz / 2
    verts = [(x, y, z) for x in (-hx, hx) for y in (-hy, hy) for z in (-hz, hz)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
             (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return mesh_from_pydata(name, verts, faces)


# ------------------------------------------------------------------- scene

def build_scene(args):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene

    fk = json.load(open(args.fk))
    fps = int(round(fk.get("fps", 30)))
    spm = fk.get("studs_per_meter", STUDS_PER_METER_DEFAULT)
    n_frames = len(fk["frames"])

    scene.render.fps = fps
    scene.frame_start = 1

    # ---- BVH import -> animated armature ------------------------------
    print("Importing BVH: %s" % args.bvh)
    bpy.ops.import_anim.bvh(
        filepath=args.bvh, global_scale=0.01, frame_start=1,
        use_fps_scale=False, update_scene_fps=True)
    arm = bpy.context.active_object
    if arm is None or arm.type != "ARMATURE":
        fail("BVH import did not produce an armature")
    scene.render.fps = fps  # keep clip fps authoritative
    act = arm.animation_data.action
    bvh_frames = int(act.frame_range[1])
    print("BVH frames: %d  FK frames: %d  fps: %d" % (bvh_frames, n_frames, fps))
    scene.frame_end = min(bvh_frames, n_frames)

    # ---- mannequin ------------------------------------------------------
    mat_amber = make_material("kimodo_amber", AMBER)
    for bone in arm.data.bones:
        radius = BONE_RADII.get(bone.name)
        is_head = bone.name == HEAD_BONE
        if radius is None and not is_head:
            continue
        length = max(bone.length, 0.05)
        if is_head:
            me = make_uv_sphere_mesh("mq_Head", HEAD_RADIUS)
            offset = Vector((0.0, -length * 0.5, 0.0))
        else:
            me = make_capsule_mesh("mq_" + bone.name, radius, length)
            offset = Vector((0.0, 0.0, 0.0))
        obj = link(bpy.data.objects.new("mq_" + bone.name, me))
        obj.data.materials.append(mat_amber)
        obj.parent = arm
        obj.parent_type = "BONE"
        obj.parent_bone = bone.name
        obj.location = offset  # bone parenting attaches at the bone TAIL

    # ---- R15: real rig meshes (plus cylinder head) ----------------------
    mat_grey = make_material("r15_grey", R15_GREY)
    r15_objs = {}
    for part in fk["parts"]:
        if part == "HumanoidRootPart":
            continue  # invisible in Roblox
        if part == "Head":
            # classic built-in Head SpecialMesh: rounded vertical cylinder
            me = make_cylinder_mesh("r15_Head", HEAD_CYL_DIAMETER / 2 / spm,
                                    HEAD_CYL_HEIGHT / spm)
            obj = link(bpy.data.objects.new("r15_Head", me))
            obj.data.materials.append(mat_grey)
            bev = obj.modifiers.new("bevel", "BEVEL")
            bev.width = 0.045
            bev.segments = 3
            bev.limit_method = "ANGLE"
        else:
            obj = import_part_obj(part, spm)
            obj.data.materials.append(mat_grey)
        obj.rotation_mode = "QUATERNION"
        r15_objs[part] = obj

    print("Keyframing %d R15 parts x %d frames ..." % (len(r15_objs), n_frames))
    for fi, frame in enumerate(fk["frames"]):
        fnum = fi + 1
        for part, obj in r15_objs.items():
            px, py, pz, qw, qx, qy, qz = frame[part]
            p = rbx_pos_to_blender((px, py, pz), spm)
            p.x += R15_X_OFFSET
            q = Q_CONV @ Quaternion((qw, qx, qy, qz))
            obj.location = p
            obj.rotation_quaternion = q
            obj.keyframe_insert("location", frame=fnum)
            obj.keyframe_insert("rotation_quaternion", frame=fnum)

    # ---- ground + world -------------------------------------------------
    gme = make_box_mesh("ground", 600.0, 600.0, 0.04)
    ground = link(bpy.data.objects.new("ground", gme))
    ground.location = (0, 0, -0.02)
    ground.data.materials.append(make_material("ground_grey", GROUND_GREY))

    world = bpy.data.worlds.new("world")
    world.color = BACKGROUND
    scene.world = world

    # ---- camera target: midpoint of the two hips, sampled per frame ----
    mids = []
    for f in range(scene.frame_start, scene.frame_end + 1):
        scene.frame_set(f)
        hip = arm.matrix_world @ arm.pose.bones["Hips"].head
        lt = fk["frames"][f - 1]["LowerTorso"]
        r15 = rbx_pos_to_blender(lt[:3], spm)
        r15.x += R15_X_OFFSET
        mids.append(((hip.x + r15.x) / 2, (hip.y + r15.y) / 2))
    # moving-average smoothing so the camera does not bob with the gait
    half = TRACK_SMOOTH // 2
    smoothed = []
    for i in range(len(mids)):
        lo, hi = max(0, i - half), min(len(mids), i + half + 1)
        smoothed.append((sum(m[0] for m in mids[lo:hi]) / (hi - lo),
                         sum(m[1] for m in mids[lo:hi]) / (hi - lo)))

    target = link(bpy.data.objects.new("cam_target", None))
    for f, (mx, my) in enumerate(smoothed, start=scene.frame_start):
        target.location = (mx, my, 1.0)
        target.keyframe_insert("location", frame=f)

    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 42.0
    cam = link(bpy.data.objects.new("cam", cam_data))
    cam.parent = target
    cam.location = (CAM_DIST * math.sin(CAM_ANGLE),
                    -CAM_DIST * math.cos(CAM_ANGLE), CAM_HEIGHT)
    con = cam.constraints.new("TRACK_TO")
    con.target = target
    con.track_axis = "TRACK_NEGATIVE_Z"
    con.up_axis = "UP_Y"
    scene.camera = cam

    # ---- labels: text objects fixed to the camera ----------------------
    # (the bundled imageio-ffmpeg lacks drawtext, so labels are baked here)
    mat_label = make_material("label_blue", R15_BLUE)
    for text, x, alignx, mat in (
            ("KIMODO SOURCE", -0.82, "LEFT", mat_amber),
            ("ROBLOX R15 (RETARGETED)", 0.82, "RIGHT", mat_label)):
        cu = bpy.data.curves.new("label", "FONT")
        cu.body = text
        cu.size = 0.075
        cu.align_x = alignx
        obj = link(bpy.data.objects.new("label_" + alignx, cu))
        obj.data.materials.append(mat)
        obj.parent = cam
        obj.location = (x, -0.46, -2.0)

    # ---- render settings (Workbench, studio look) -----------------------
    scene.render.engine = "BLENDER_WORKBENCH"
    try:
        w, h = (int(v) for v in args.size.lower().split("x"))
    except ValueError:
        fail("--size must look like 1280x720")
    scene.render.resolution_x = w
    scene.render.resolution_y = h
    scene.render.film_transparent = False
    scene.display.render_aa = "8"
    scene.display.light_direction = Vector((0.35, -0.35, 0.87)).normalized()
    scene.display.shadow_shift = 0.06
    sh = scene.display.shading
    sh.light = "STUDIO"
    sh.color_type = "MATERIAL"
    sh.show_shadows = True
    sh.shadow_intensity = 0.45
    sh.show_cavity = True
    sh.cavity_type = "WORLD"
    sh.cavity_ridge_factor = 0.4
    sh.cavity_valley_factor = 0.6
    sh.show_object_outline = False

    scene.render.image_settings.file_format = "PNG"
    return scene


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isfile(args.bvh):
        fail("BVH not found: %s" % args.bvh)
    if not os.path.isfile(args.fk):
        fail("FK json not found: %s" % args.fk)

    scene = build_scene(args)
    scene.render.filepath = os.path.join(args.output_dir, "frame_")

    if args.stills:
        for f in (int(v) for v in args.stills.split(",")):
            scene.frame_set(f)
            scene.render.filepath = os.path.join(
                args.output_dir, "still_%04d.png" % f)
            bpy.ops.render.render(write_still=True)
            print("wrote %s" % scene.render.filepath)
    else:
        print("Rendering frames %d..%d at %dfps ..."
              % (scene.frame_start, scene.frame_end, scene.render.fps))
        bpy.ops.render.render(animation=True)
    print("DONE")


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # ensure non-zero exit for blender -b
        import traceback
        traceback.print_exc()
        sys.exit(1)
