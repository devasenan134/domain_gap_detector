import blenderproc as bproc  # must be the first import (BlenderProc requirement)

"""Render one synthetic YCB-V variant and write it straight to YOLO format.

Usage (from the repo root):
    blenderproc run blender/generate.py blender/configs/baseline.yaml

Output:
    data/yolo_dataset/<variant>/
        images/{train,val}/SSSS_FF.png
        labels/{train,val}/SSSS_FF.txt   # cls cx cy w h, normalized
        dataset.yaml

Boxes come from the instance segmentation map, i.e. they are *visible* boxes,
matching the `bbox_visib` used for the real test set in prep_ycbv_test.ipynb.
Each scene is seeded independently, so a killed run can be resumed and
already-rendered scenes are skipped.
"""

import argparse
import os

import numpy as np
import yaml
from PIL import Image

# BOP obj_id -> YOLO class id (same mapping as the real test set)
TARGET_IDS = {1: 0, 2: 1, 4: 2, 5: 3, 10: 4, 12: 5, 15: 6, 17: 7}
CLASS_NAMES = [
    "002_master_chef_can",
    "003_cracker_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "011_banana",
    "021_bleach_cleanser",
    "035_power_drill",
    "037_scissors",
]

parser = argparse.ArgumentParser()
parser.add_argument("config", help="variant YAML in blender/configs/")
parser.add_argument("--bop-path", default="data", help="folder that contains ycbv/")
parser.add_argument("--out", default="data/yolo_dataset", help="variants are written under here")
parser.add_argument("--num-scenes", type=int, help="override num_scenes (quick tests)")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)
if args.num_scenes is not None:
    cfg["num_scenes"] = args.num_scenes

out_dir = os.path.abspath(os.path.join(args.out, cfg["variant"]))
for split in ("train", "val"):
    os.makedirs(os.path.join(out_dir, "images", split), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "labels", split), exist_ok=True)

ycbv_path = os.path.abspath(os.path.join(args.bop_path, "ycbv"))


def uniform(lo_hi):
    """Sample from [lo, hi]; works for scalars and vectors (lo == hi pins the value)."""
    lo, hi = np.asarray(lo_hi[0], dtype=float), np.asarray(lo_hi[1], dtype=float)
    return np.random.uniform(lo, hi)


# ---------------------------------------------------------------- scene setup
bproc.init()

target_objs = bproc.loader.load_bop_objs(
    bop_dataset_path=ycbv_path, mm2m=True, obj_ids=list(TARGET_IDS)
)
bproc.loader.load_bop_intrinsics(bop_dataset_path=ycbv_path)  # 640x480, real YCB-V camera

for obj in target_objs:
    obj.set_shading_mode("auto")
    obj.set_cp("category_id", TARGET_IDS[obj.get_cp("category_id")] + 1)  # 0 = background
    obj.hide(True)

# Open box: floor + 4 walls
room_planes = [
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]),
    bproc.object.create_primitive("PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]),
]
for plane in room_planes:
    plane.enable_rigidbody(False, collision_shape="BOX", mass=1.0, friction=100.0,
                           linear_damping=0.99, angular_damping=0.99)

light_plane = bproc.object.create_primitive("PLANE", scale=[3, 3, 1], location=[0, 0, 10])
light_plane.set_name("light_plane")
light_plane_material = bproc.material.create("light_material")
light_point = bproc.types.Light()

bg = cfg["background"]
if bg["type"] == "cc_textures":
    cc_textures = bproc.loader.load_ccmaterials(os.path.abspath(bg["cc_textures_path"]))
else:
    plain_material = bproc.material.create("plain_background")
    plain_material.set_principled_shader_value("Base Color", [*bg["color"], 1.0])
    plain_material.set_principled_shader_value("Roughness", 0.8)
    for plane in room_planes:
        plane.replace_materials(plain_material)

bproc.renderer.set_max_amount_of_samples(cfg["render"]["samples"])
if not cfg["render"]["denoiser"]:
    bproc.renderer.set_denoiser(None)
bproc.renderer.enable_segmentation_output(map_by=["instance", "category_id"],
                                          default_values={"category_id": 0})


def sample_pose_func(obj):
    lo = np.random.uniform([-0.15, -0.15, 0.0], [-0.1, -0.1, 0.0])
    hi = np.random.uniform([0.1, 0.1, 0.15], [0.15, 0.15, 0.25])
    obj.set_location(np.random.uniform(lo, hi))
    obj.set_rotation_euler(bproc.sampler.uniformSO3())


def yolo_lines(segmap, attribute_map, min_pixels):
    h, w = segmap.shape
    lines = []
    for inst in attribute_map:
        if inst["category_id"] == 0:
            continue
        ys, xs = np.nonzero(segmap == inst["idx"])
        if len(xs) < min_pixels:
            continue
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        lines.append(f"{inst['category_id'] - 1} {cx:.6f} {cy:.6f} {(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return lines


# ---------------------------------------------------------------- render loop
num_scenes, per_scene = cfg["num_scenes"], cfg["images_per_scene"]
num_val = int(round(num_scenes * cfg["val_fraction"]))
cam, light, objs_cfg = cfg["camera"], cfg["lighting"], cfg["objects"]

for scene in range(num_scenes):
    split = "val" if scene >= num_scenes - num_val else "train"
    last = os.path.join(out_dir, "labels", split, f"{scene:04d}_{per_scene - 1:02d}.txt")
    if os.path.exists(last):
        print(f"scene {scene} already rendered, skipping")
        continue
    np.random.seed(cfg["seed"] * 100_000 + scene)

    n = np.random.randint(objs_cfg["per_scene"][0], objs_cfg["per_scene"][1] + 1)
    scene_objs = list(np.random.choice(target_objs, size=min(n, len(target_objs)), replace=False))

    s = objs_cfg["material_randomization"]
    for obj in scene_objs:
        if s > 0:
            mat = obj.get_materials()[0]
            mat.set_principled_shader_value("Roughness", np.random.uniform(0.5 - 0.5 * s, 0.5 + 0.5 * s))
            mat.set_principled_shader_value("Specular IOR Level", np.random.uniform(0.5 - 0.5 * s, 0.5 + 0.5 * s))
        obj.enable_rigidbody(True, mass=1.0, friction=100.0, linear_damping=0.99, angular_damping=0.99)
        obj.hide(False)

    light_plane_material.make_emissive(
        emission_strength=uniform(light["ceiling_strength"]),
        emission_color=[*uniform(light["ceiling_color"]), 1.0],
    )
    light_plane.replace_materials(light_plane_material)
    light_point.set_energy(uniform(light["point_light_energy"]))
    light_point.set_color(uniform(light["point_light_color"]))
    light_point.set_location(bproc.sampler.shell(center=[0, 0, 0], radius_min=1, radius_max=1.5,
                                                 elevation_min=5, elevation_max=89))

    if bg["type"] == "cc_textures":
        texture = np.random.choice(cc_textures)
        for plane in room_planes:
            plane.replace_materials(texture)

    bproc.object.sample_poses(objects_to_sample=scene_objs, sample_pose_func=sample_pose_func, max_tries=1000)
    bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=3, max_simulation_time=10,
                                                      check_object_interval=1, substeps_per_frame=20,
                                                      solver_iters=25)
    bvh_tree = bproc.object.create_bvh_tree_multi_objects(scene_objs)

    poses, tries = 0, 0
    while poses < per_scene and tries < 10_000:
        tries += 1
        location = bproc.sampler.shell(center=[0, 0, 0], radius_min=cam["radius"][0], radius_max=cam["radius"][1],
                                       elevation_min=cam["elevation_deg"][0], elevation_max=cam["elevation_deg"][1])
        poi = bproc.object.compute_poi(scene_objs)
        inplane = np.deg2rad(np.random.uniform(-cam["inplane_rot_deg"], cam["inplane_rot_deg"]))
        rotation = bproc.camera.rotation_from_forward_vec(poi - location, inplane_rot=inplane)
        cam2world = bproc.math.build_transformation_mat(location, rotation)
        if not bproc.camera.perform_obstacle_in_view_check(cam2world, {"min": 0.3}, bvh_tree):
            continue
        visible = set(bproc.camera.visible_objects(cam2world)) & set(scene_objs)
        if len(visible) < cam["min_visible_objects"]:
            continue
        bproc.camera.add_camera_pose(cam2world, frame=poses)
        poses += 1
    bproc.utility.set_keyframe_render_interval(frame_start=0, frame_end=poses)

    data = bproc.renderer.render()

    for frame in range(poses):
        name = f"{scene:04d}_{frame:02d}"
        Image.fromarray(data["colors"][frame]).save(os.path.join(out_dir, "images", split, name + ".png"))
        lines = yolo_lines(data["instance_segmaps"][frame], data["instance_attribute_maps"][frame],
                           cfg["labels"]["min_visible_pixels"])
        with open(os.path.join(out_dir, "labels", split, name + ".txt"), "w") as f:
            f.write("".join(line + "\n" for line in lines))

    for obj in scene_objs:
        obj.disable_rigidbody()
        obj.hide(True)
    print(f"scene {scene + 1}/{num_scenes} done ({poses} images, {split})")

# ---------------------------------------------------------------- dataset.yaml
with open(os.path.join(out_dir, "dataset.yaml"), "w") as f:
    yaml.safe_dump({
        "path": out_dir,
        "train": "images/train",
        "val": "images/val",
        "test": "../test/images",  # real YCB-V test set from prep_ycbv_test.ipynb
        "names": dict(enumerate(CLASS_NAMES)),
    }, f, sort_keys=False)
print(f"done -> {out_dir}")
