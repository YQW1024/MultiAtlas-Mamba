# -*- coding: utf-8 -*-

import os
import subprocess

atlas_path = "/media/UG2/yqw/T1/Atlas/sa/shen_1mm_268_parcellation_mni.nii.gz"#tupu

hcp_root = "/media/UG2/yqw/T1/Caff/"

for subject in os.listdir(hcp_root):
    sub_path = os.path.join(hcp_root, subject)

    t1_path = os.path.join(sub_path, "t1w.nii.gz")

    mni_path = os.path.join(sub_path, "MNI_new")

    warp_file = os.path.join(mni_path, "Atlas2SubWarpWarp.nii")
    affine_file = os.path.join(mni_path, "Atlas2SubWarpAffine.txt")

    output_image = os.path.join(sub_path, "shen_1mm_268_parcellation_mni.nii.gz")#tupu

    if not (os.path.exists(atlas_path) and
            os.path.exists(t1_path) and
            os.path.exists(warp_file) and
            os.path.exists(affine_file)):
        print(f"[Skip] Missing T1 or transform files for {subject}, skipping...")
        continue

    warp_cmd = [
        "WarpImageMultiTransform", "3",
        atlas_path,
        output_image,
        "-R", t1_path,
        warp_file,
        affine_file,
        "--use-NN"
    ]

    try:
        subprocess.run(warp_cmd, check=True)
        print(f"[Done] {subject}: AICHA mapped to T1 space.")
    except subprocess.CalledProcessError as e:
        print(f"[Error] {subject}: warp failed {e}")
