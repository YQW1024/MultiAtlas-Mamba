# -*- coding: utf-8 -*-
import numpy as np
import os
from batchgenerators.utilities.file_and_folder_operations import subfiles, join


base_dir = '/data/yqw/U-Mamba/nnUNet_preprocessed/Dataset002_T1_aparc2009/nnUNetPlans_3d_fullres/'
second_dir = '/data/yqw/U-Mamba/nnUNet_preprocessed/Dataset001_T1_wmparc/nnUNetPlans_3d_fullres/'

def merge_preprocessed_data_standard():

    files = subfiles(base_dir, suffix='.npz')

    if len(files) == 0:
        print("No .npz files found, checking for .npy...")
        files = subfiles(base_dir, suffix='.npy')

    for f in files:
        file_name = os.path.basename(f)
        second_file_path = join(second_dir, file_name)
        
        if not os.path.exists(second_file_path):
            print(f"Warning: {file_name} not found in second directory, skipping.")
            continue

        data_base = np.load(f, allow_pickle=True)
        data_second = np.load(second_file_path, allow_pickle=True)

        if f.endswith('.npz'):
            image = data_base['data']
            label_184 = data_base['seg']

            label_90 = data_second['seg']
        else:
            arr_base = data_base
            arr_second = data_second
            image = arr_base[0:1]
            label_184 = arr_base[1:2]
            label_90 = arr_second[1:2]

        merged_seg = np.concatenate([label_184, label_90], axis=0)

        if f.endswith('.npz'):
            np.savez(f, data=image, seg=merged_seg)
        else:
            np.savez(f.replace('.npy', '.npz'), data=image, seg=merged_seg)
            
        print(f"Successfully merged: {file_name}. Seg shape: {merged_seg.shape}")

if __name__ == '__main__':
    merge_preprocessed_data_standard()