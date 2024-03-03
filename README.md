# SCP-Nano: Deep Learning Powered Imaging of Nanocarriers Across Entire Mouse Bodies at Single-Cell Resolution
![pipeline](./images/pipeline.png)

Overview of SCP-Nano for whole-body image analysis of nanocarriers at single-cell resolution 

## Table of contents
1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Pipeline workflow](#pipeline-workflow)

## Introduction
SCP-Nano (Single Cell Precision Nanocarrier Identification pipeline) is a pipeline combining whole-mouse body imaging at cellular resolution with deep learning to reveal the targeting of tens of millions of cells by nanocarriers. This repository contains the source code to run SCP-Nano from scratch.


## Requirements
### system requirements
Linux system with GPU (at least 10 GB GPU RAM) and CPU (at least 20 cores), and with 400 GB RAM.

### data requirements
* Raw image data saved as a series of 16-bit TIFF files, one per z-plane. 
* Organ annotation of the raw image data saves as a series of 8-bit TIFF files, one per z-plane.
* Text file storing organ annotation label details following such a template: `organ label value: organ name`.  An example is provided [here](./example/organ_keys.txt). 
* Organize Raw image TIFF series of nanoparticle signal channel (C01), organ annotation TIFF series (organ_mask) and organ label detail txt file (organ_keys.txt) as following:
   ```
    dir_wholebody_data/
    ├── C01/
    │   ├──C01_Z0000.tif
    │   ├──C01_Z0001.tif
    │   ├──...
    ├── organ_mask/
    │   ├──slice_0000.tif
    │   ├──slice_0001.tif
    │   ├──...
    ├── organ_keys.txt
    ```
    The TIFF series' prefixes don't have to match, but ensure that the z slice ID starts from 0000.

## Installation
1. Install [Anaconda](https://docs.anaconda.com/free/anaconda/install/linux/) to create and control virtual environments. 
2. Install [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) to utilize the power GPUs for deep learning 
3. Install Python 3.9 or higher version by Anaconda.
   ```
    conda create -n your_env python=3.9
	conda activate your_env
	```
4. Clone this repository into a local directory in your device by `git clone https://github.com/erturklab/SCP-Nano.git dir_you_save`    
5. Install libraries for general data processing by CPUs:
   ```
     cd dir_you_save
     pip install -r requirements.txt
	```
6. Install [pytorch](https://pytorch.org/get-started/locally/), and then other libraries for deep learing segmentation using GPUs:
   ```     
     cd dir_you_save/2_DL_segmentation_pred
     pip install -e .
	``` 

## Pipeline workflow