# MultiAtlas-Mamba

**Unified Parcellation of Label-Unavailable Multi-Modal Atlases via Cross-Atlas Synergistic Learning**


## Overview
This repository contains the official implementation of **MultiAtlas-Mamba**, a unified, registration-free framework for simultaneous multi-atlas brain parcellation.

Unlike traditional methods that rely on time-consuming pairwise registration, our method uses a **single T1-weighted MRI** to simultaneously predict multiple heterogeneous atlases (e.g., WMPARC, APARC2009, AICHA, Brainnetome). It features a shared **U-Mamba** encoder to capture long-range anatomical dependencies and introduces a **Cross-Atlas Synergistic Learning** strategy. This allows the model to leverage strong supervision from anatomical ground truths to rectify noise in label-unavailable (pseudo-labeled) functional tasks.

![Overview of MultiAtlas-Mamba](image/overview.png)
*(Note: Please ensure you have an 'image' folder and put your 'overview.png' diagram inside it)*

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
pip install "causal-conv1d>=1.2.0"
pip install mamba-ssm --no-cache-dir
```
```bash
cd umamba
pip install -e .
```
