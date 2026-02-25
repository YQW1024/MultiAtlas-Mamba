# MultiAtlas-Mamba

**Unified Parcellation of Label-Unavailable Multi-Modal Atlases via Cross-Atlas Synergistic Learning**


## Overview
This repository contains the official implementation of **MultiAtlas-Mamba**, a unified, registration-free framework for simultaneous multi-atlas brain parcellation.

Unlike traditional methods that rely on time-consuming pairwise registration, our method uses a **single T1-weighted MRI** to simultaneously predict multiple heterogeneous atlases (e.g., WMPARC, APARC2009, AICHA, Brainnetome). It features a shared **U-Mamba** encoder to capture long-range anatomical dependencies and introduces a **Cross-Atlas Synergistic Learning** strategy. This allows the model to leverage strong supervision from anatomical ground truths to rectify noise in label-unavailable (pseudo-labeled) functional tasks.

![Overview of MultiAtlas-Mamba](image/overview.png)

## Create a Virtual Environment
```bash
conda create -n MultiAtlasMamba_env python=3.10 -y
conda activate MultiAtlasMamba_env
```
## Install PyTorch 2.0.1
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```
## Install Other Dependencies
```bash
pip install antspyx nibabel SimpleITK
pip install "causal-conv1d>=1.2.0"
pip install mamba-ssm --no-cache-dir
```
```bash
cd umamba
pip install -e .
```
## Pseudo-label Generation
- **Spatial Alignment:** `T1_ants.py`  
  Performs non-linear registration to align the standard atlas to the individual T1 space.
- **Label Mapping:** `T1_ants_seg.py`  
  Maps the atlas labels to the subject space using the calculated deformation field to generate pseudo-labels.

## Dataset Format
The dataset should be organized in the [`nnU-Net`](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_format.md) format.
To verify the dataset integrity, run:
```bash
nnUNetv2_plan_and_preprocess -d <DATASET_NAME> --verify_dataset_integrity
```
- **Multi-Atlas Label Integration:** `atlas_integration.py`  
  Merges multiple individual atlas labels into a unified format for joint multi-task training. In our Multi-Task framework, labels are integrated into 6 channels. By default, the 2nd and 3rd channels are treated as Strongly-Supervised tasks using ground truths (GT), while others use registration-based pseudo-labels.

## Training
```bash
nnUNetv2_train <DATASET_NAME> 3d_fullres all -tr nnUNetTrainerUMambaBot
```
## Prediction
```bash
nnUNetv2_predict -i <INPUT_FOLDER> -o <OUTPUT_FOLDER> -d <DATASET_NAME> -c 3d_fullres -f all -tr nnUNetTrainerUMambaBot --disable_tta
```
