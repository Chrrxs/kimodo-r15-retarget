#!/usr/bin/env python3
"""Download the Kimodo rig's MeshPart assets and convert them to OBJ.

The rig (classic blocky R15 bundle) uses public Roblox mesh assets in the
binary `version 2.00` format:

    "version 2.00\\n"
    uint16 headerSize (12)
    uint8  vertexSize (36 or 40)
    uint8  faceSize   (12)
    uint32 numVerts
    uint32 numFaces
    verts: [px py pz nx ny nz u v (float32) + vertexSize-32 extra bytes] * n
    faces: [uint32 a, uint32 b, uint32 c] * n   (0-based)

Vertices are in the part's local space (studs, y-up, mesh scale 1:1 for this
bundle). Writes one OBJ per part into retarget/kimodo_rig/.

Usage: python3 retarget/fetch_roblox_meshes.py [--geometry retarget/r15_rig_geometry_kimodo.json]
"""
import argparse
import gzip
import json
import struct
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ASSET_URL = "https://assetdelivery.roblox.com/v1/asset/?id={}"


def parse_mesh_v2(data):
    nl = data.index(b"\n")
    version = data[:nl].decode()
    if not version.startswith("version 2.00"):
        raise ValueError(f"unsupported mesh format: {version!r}")
    off = nl + 1
    header_size, vertex_size, face_size = struct.unpack_from("<HBB", data, off)
    num_verts, num_faces = struct.unpack_from("<II", data, off + 4)
    if header_size != 12:
        raise ValueError(f"unexpected v2 header size {header_size}")
    off += header_size
    verts = []
    normals = []
    for i in range(num_verts):
        px, py, pz, nx, ny, nz = struct.unpack_from("<6f", data, off + i * vertex_size)
        verts.append((px, py, pz))
        normals.append((nx, ny, nz))
    off += num_verts * vertex_size
    faces = []
    for i in range(num_faces):
        a, b, c = struct.unpack_from("<3I", data, off + i * face_size)
        faces.append((a, b, c))
    return verts, normals, faces


def write_obj(path, verts, normals, faces):
    with open(path, "w") as fh:
        fh.write("# converted from Roblox mesh v2.00\n")
        for v in verts:
            fh.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in normals:
            fh.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        for a, b, c in faces:
            fh.write(f"f {a+1}//{a+1} {b+1}//{b+1} {c+1}//{c+1}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry", default=str(HERE / "r15_rig_geometry_kimodo.json"))
    args = ap.parse_args()

    geometry = json.loads(Path(args.geometry).read_text())
    out_dir = HERE / "kimodo_rig"
    out_dir.mkdir(exist_ok=True)

    for part, asset_id in geometry["meshes"].items():
        out = out_dir / f"{part}.obj"
        if out.exists():
            print(f"{part}: already fetched")
            continue
        req = urllib.request.Request(
            ASSET_URL.format(asset_id), headers={"User-Agent": "kimodo-pipeline/1.0"}
        )
        data = urllib.request.urlopen(req, timeout=30).read()
        if data[:2] == b"\x1f\x8b":  # CDN gzips regardless of Accept-Encoding
            data = gzip.decompress(data)
        verts, normals, faces = parse_mesh_v2(data)
        write_obj(out, verts, normals, faces)
        print(f"{part}: {len(verts)} verts, {len(faces)} tris -> {out.name}")


if __name__ == "__main__":
    main()
