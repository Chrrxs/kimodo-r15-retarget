SHELL := /bin/bash

.PHONY: test retarget compare compare-blender fetch-meshes

RETARGET_INPUT ?=
RETARGET_ARGS ?=
BLENDER ?= blender
COMPARE_SIZE ?= 1280x720

test:
	python3 retarget/tests/test_retarget.py

# BVH -> Roblox R15 KeyframeSequence (.rbxmx), written next to the BVH.
retarget:
	@test -n "$(RETARGET_INPUT)" || (echo 'Usage: make retarget RETARGET_INPUT=path/to/motion.bvh'; exit 2)
	python3 retarget/bvh_to_r15.py $(RETARGET_INPUT) --output $(basename $(RETARGET_INPUT)).rbxmx $(RETARGET_ARGS)

# Fast wireframe side-by-side comparison video (<name>_compare.mp4).
compare:
	@test -n "$(RETARGET_INPUT)" || (echo 'Usage: make compare RETARGET_INPUT=path/to/motion.bvh'; exit 2)
	@test -x .venv/bin/python || python3 -m venv .venv
	@.venv/bin/python -c 'import PIL, imageio_ffmpeg' 2>/dev/null || .venv/bin/pip install --quiet pillow imageio-ffmpeg
	.venv/bin/python retarget/compare_video.py $(RETARGET_INPUT) $(RETARGET_ARGS)

# Download the classic blocky R15 meshes from Roblox's CDN (not bundled).
fetch-meshes:
	python3 retarget/fetch_roblox_meshes.py

# Blender-rendered side-by-side comparison (motion_compare_blender.mp4).
# Needs Blender 4.2 LTS: override with BLENDER=/path/to/blender.
compare-blender: fetch-meshes
	@test -n "$(RETARGET_INPUT)" || (echo 'Usage: make compare-blender RETARGET_INPUT=path/to/motion.bvh'; exit 2)
	@test -x .venv/bin/python || python3 -m venv .venv
	@.venv/bin/python -c 'import imageio_ffmpeg' 2>/dev/null || .venv/bin/pip install --quiet imageio-ffmpeg
	BLENDER=$(BLENDER) bash retarget/blender_compare_run.sh $(RETARGET_INPUT) $(COMPARE_SIZE)
