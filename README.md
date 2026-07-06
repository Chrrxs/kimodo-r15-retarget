# kimodo-r15-retarget

Convert [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) text-to-motion
output into Roblox R15 animations.

Kimodo exports BVH on the SOMA `somaskel77` skeleton (30 fps, centimeters,
standard T-pose rest). This tool retargets those files onto the R15 rig and
writes a `KeyframeSequence` as `.rbxmx` for Roblox Studio.

Core converter is Python 3 standard library only.

## Quick start

```bash
# generate a motion with kimodo:
kimodo_gen "A person walks forward confidently." \
    --model Kimodo-SOMA-RP-v1.1 --bvh --bvh_standard_tpose --output motion

# retarget to R15:
python3 retarget/bvh_to_r15.py motion.bvh --output motion.rbxmx --name walk-forward
```

In Studio: right-click in the Explorer, **Insert From File**, select the
`.rbxmx`. Put the `KeyframeSequence` under an `AnimSaves` model on an R15
rig, load it in the Animation Editor, and publish.

## CLI

```text
python3 retarget/bvh_to_r15.py input.bvh --output out.rbxmx
    [--scale-studs-per-meter 3.571]      # root-motion scale
    [--fps N]                            # resample (default: source rate)
    [--keyframe-reduction-epsilon 0.5]   # degrees; 0 disables
    [--loop]
    [--priority Action]                  # Idle|Movement|Action|Action2..4|Core
    [--forward-axis z]                   # axis the BVH character faces
    [--name MyAnim]
```

Writes a `<out>.rbxmx.meta.json` sidecar with frame counts, duration, and
reduction stats.

## What it does

- Maps SOMA joints onto the 15 R15 Motor6Ds, composing collapsed chains
  (spine to Waist, neck to Neck, shoulder+arm to Shoulder). Fingers, toes,
  eyes, and jaw are dropped.
- Applies rest-pose corrections from declared bone directions, including
  SOMA's ~17 degree forward rest-neck lean.
- Converts axes (SOMA faces +Z, Roblox -Z) and units (cm to studs).
- Writes root motion onto the LowerTorso pose; the Animator ignores
  HumanoidRootPart pose translation.
- Greedy keyframe reduction against slerp interpolation.

Math details and verified behaviors: [retarget/README.md](retarget/README.md).

## QA tools

| Tool | Purpose |
|---|---|
| `compare_video.py` | wireframe side-by-side MP4 (needs pillow, imageio-ffmpeg) |
| `blender_compare.py` / `blender_compare_run.sh` | rendered side-by-side MP4 (needs Blender 4.2 LTS) |
| `bvh_to_json.py` | BVH world joint positions as JSON |
| `export_fk_json.py` | per-frame world transforms of the retargeted R15 |

```bash
make test
make compare RETARGET_INPUT=path/to/motion.bvh
make fetch-meshes
make compare-blender RETARGET_INPUT=path/to/motion.bvh BLENDER=/path/to/blender
```

The Blender comparison uses the classic blocky R15 meshes. They are Roblox
property and not bundled: `make fetch-meshes` downloads them from Roblox's
CDN into `retarget/kimodo_rig/`.

## Licensing

MIT ([LICENSE](LICENSE)). The test fixture BVH comes from the Apache-2.0
[nv-tlabs/kimodo](https://github.com/nv-tlabs/kimodo) repo (see
[NOTICE](NOTICE)). Motions you generate are subject to NVIDIA's model
licenses; published animations are subject to Roblox's terms.

## Limitations

- R15 has fewer joints than SOMA: finger/toe articulation is lost and the
  three-joint spine collapses onto one Waist joint.
- Bone twist uses the minimal-rotation convention; declare per-joint twist
  offsets in `retarget/r15.py` if an arm roll looks off.
- No IK pass, so proportion differences can cause minor foot slide.
- SOMA's ~5 degree rest shin lean is left uncorrected to keep feet flat.
