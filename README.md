# Progressive-Neural-Compression

### [Dec 15, 2023] We are actively uploading the code files for simulation and experiments. If you have any questions, please contact the authors. The network and the checkpoints are in the `demo_simulation` folder

## Introduction

This repository contains the source code and testbed setup instructions for **R. Wang, H. Liu, J. Qiu, M. Xu, R. Guerin, C. Lu, Progressive Neural Compression for Adaptive Image Offloading under Timing Constraints, IEEE Real-Time Systems Symposium ([RTSS'23](https://2023.rtss.org/)), December 2023. [[arxiv](https://arxiv.org/pdf/2310.05306.pdf)]**
![pnc_overview](assets/pnc_overview.png?raw=true)

## Quick Demo
* Install the required environment, mainly TensorFlow. We used tf.keras and the code should be compatible with most Tensorflow 2 versions >2.5, but if it raises an error please consider `tensorflow==2.8.0`.
* Put the ImageNet Val images (named as `ILSVRC2012_val_00000001.JPEG`, etc.) in `demo_simulation\val2017`. There are multiple sources to download this dataset, e.g. from [Kaggle ImageNet Object Localization Challenge](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data).
* The demo file is located at: `demo_simulation\pnc_demo_simulation.ipynb`

## Autoencoder Network
We separate out the network, training and testbed into different folders so that user can pick the components they need conveniently. 

The network definition and checkpoint loading is located at: `demo_simulation/pnc_demo_network.ipynb`

```
# Encoder
encoder_input = layers.Input(shape=(img_height, img_width, 3))
initializer = tf.keras.initializers.Orthogonal()
encoder_x = layers.Conv2D(
    16, (9, 9), 
    strides=7, 
    activation="relu", 
    padding="same", 
    kernel_initializer=initializer
)(encoder_input)
encoder_x = layers.Conv2D(
    10, (3, 3), 
    strides=1,
    activation="relu", 
    padding="same", 
    kernel_initializer=initializer,
    name='encoder_out'
)(encoder_x)
encoder_model = keras.Model(encoder_input, encoder_x,  name='enocder')
```

Simply open it with jupyter notebook and run it.


## Experimental Setup
Instructions for experimental hardware and testbed setup can be found in [testbed/](https://github.com/rickywrq/Progressive-Neural-Compression/blob/main/testbed/)