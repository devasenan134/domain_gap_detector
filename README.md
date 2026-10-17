
# Project: sim2real

## Data related:
YCB-V (YCB Video Dataset): Contains 21 household and kitchen objects with rich textures and colors (e.g., cracker boxes, mustard bottles, soup cans). It features complex real-world clutter, making it excellent for testing object recognition in messy environments.

    1. bop_toolkit from: https://github.com/thodan/bop_toolkit
    2. models from: the original dataset https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_models.zip
    3. test set from BOP`19-24 test images: https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/main/ycbv_test_bop19.zip
    

textures for scenes
blenderproc download cc_textures cc_textures/

## Synthetic data generation

One YAML per variant in `blender/configs/`. `blender/generate.py` renders it and writes YOLO labels directly
(visible boxes from the segmentation map, same as `bbox_visib` in the real test set) — no COCO step needed.

    uv run blenderproc run blender/generate.py blender/configs/baseline.yaml
    # -> data/yolo_dataset/baseline/{images,labels}/{train,val}, dataset.yaml (test: real YCB-V)

Needs `data/ycbv/models` + `data/ycbv/camera_uw.json` (ycbv_models.zip + ycbv_base.zip).
Re-running resumes: finished scenes are skipped. `--num-scenes 2` for a quick test.

macOS: BlenderProc can't install its pip packages inside the protected `Blender.app`, so copy it out once and point at the copy:

    cd ~/blender/blender-4.2.1-macos-arm64 && cp -R Blender.app blender_unbundled && ln -s Contents/Resources/4.2 blender_unbundled/4.2
    uv run blenderproc run --custom-blender-path ~/blender/blender-4.2.1-macos-arm64/blender_unbundled blender/generate.py blender/configs/baseline.yaml