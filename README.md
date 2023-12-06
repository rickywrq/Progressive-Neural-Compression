# Progressive-Neural-Compression

### [Dec 5, 2023] We are actively uploading the code files for simulation and experiments. If you have any questions, please contact the authors.

## Introduction

This repository contains the source code and testbed setup instructions for **R. Wang, H. Liu, J. Qiu, M. Xu, R. Guerin, C. Lu, [Progressive Neural Compression for Adaptive Image Offloading under Timing Constraints](https://arxiv.org/pdf/2310.05306.pdf), IEEE Real-Time Systems Symposium ([RTSS'23](https://2023.rtss.org/)), December 2023.**
![pnc_overview](assets\pnc_overview.png?raw=true)

## Quick Demo
* Install the required environment, mainly TensorFlow. We tried to use tf.keras and make the code compatible for most Tensorflow versions, but if it raises an error please consider `tensorflow==2.8.0`.
* Put the ImageNet Val images (named as `ILSVRC2012_val_00000001.JPEG`, etc.) in `demo_simulation\val2017`. There are multiple sources to download this dataset, e.g. from [Kaggle ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
* The demo file is located at: `demo_simulation\pnc_demo_simulation.ipynb`

## Experimental Setup
