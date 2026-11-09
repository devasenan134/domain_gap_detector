"""Download a small subset of ambientCG materials for the cc_textures background.

    uv run python blender/download_textures.py cc_textures 100

`blenderproc download cc_textures` fetches every material at 2K (about 55 GB). This takes the
first N at 1K-JPG instead. BlenderProc's loader only looks for files named *_2K-JPG_*, so the
1K files are renamed to match; they are still 1K images, which is plenty for 640x480 renders.
"""

import io
import sys
import zipfile
from pathlib import Path

import requests

out, n = Path(sys.argv[1]), int(sys.argv[2])
headers = {"User-Agent": "Mozilla/5.0"}  # the server rejects the default one
# Same exclusions as BlenderProc's own downloader (signs, roadlines, 3D props, ...)
skip = ("sign", "roadlines", "manhole", "backdrop", "foliage", "treeend", "treestump", "3dbread",
        "3dapple", "flowerset", "foodsteps", "pineneedles", "grate", "pavingedge", "painting",
        "rockbrush", "wrinklesbrush", "sticker", "3drock")

got, offset = 0, 0
while got < n:
    page = requests.get("https://ambientcg.com/api/v2/full_json", headers=headers, timeout=30,
                        params={"include": "downloadData", "limit": 100, "offset": offset, "type": "material"}).json()
    assets = page.get("foundAssets", [])
    if not assets:
        break
    offset += 100
    for asset in assets:
        name = asset["assetId"]
        if got >= n or name.lower().startswith(skip):
            continue
        downloads = asset["downloadFolders"]["default"]["downloadFiletypeCategories"]["zip"]["downloads"]
        link = next((d["downloadLink"] for d in downloads if d["attribute"] == "1K-JPG"), None)
        if link is None:
            continue
        dest = out / name
        if not (dest / f"{name}_2K-JPG_Color.jpg").exists():
            archive = zipfile.ZipFile(io.BytesIO(requests.get(link, headers=headers, timeout=60).content))
            dest.mkdir(parents=True, exist_ok=True)
            for member in archive.namelist():
                if member.endswith(".jpg"):  # basename only, so nothing can be written outside dest
                    (dest / Path(member).name.replace("_1K-JPG_", "_2K-JPG_")).write_bytes(archive.read(member))
        got += 1
        print(f"{got}/{n} {name}")
