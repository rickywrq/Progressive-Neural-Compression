# Progressive-Neural-Compression

### [Feb, 2024] We are actively uploading the code files for training. If you have any questions, please contact the authors. The network and the checkpoints are in the `demo_simulation` folder

## Introduction

This repository contains the source code and testbed setup instructions for **R. Wang, H. Liu, J. Qiu, M. Xu, R. Guerin, C. Lu, Progressive Neural Compression for Adaptive Image Offloading under Timing Constraints, IEEE Real-Time Systems Symposium ([RTSS'23](https://2023.rtss.org/)), December 2023. [[arXiv](https://arxiv.org/pdf/2310.05306.pdf)] [[RTSS 2023 Proceedings](https://doi.ieeecomputersociety.org/10.1109/RTSS59052.2023.00020)]**
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

The encoder model `demo_simulation/saved_tflite_models_demo/best_encoder_tuned_model_uint8.tflite` can be visualized by [Netron](https://netron.app/).

![pnc_encoder](assets/pnc_encoder_netron.png?raw=true)
An example visualization of the encoder model.

## Experimental Setup
Instructions for experimental hardware and testbed setup can be found in [testbed/](https://github.com/rickywrq/Progressive-Neural-Compression/blob/main/testbed/)


## Citation

If you find our paper useful, please consider citing this in your publication.

```bibtex
@INPROCEEDINGS {10405983,
author = {R. Wang and H. Liu and J. Qiu and M. Xu and R. Guerin and C. Lu},
booktitle = {2023 IEEE Real-Time Systems Symposium (RTSS)},
title = {Progressive Neural Compression for Adaptive Image Offloading Under Timing Constraints},
year = {2023},
volume = {},
issn = {},
pages = {118-130},
abstract = {IoT devices are increasingly the source of data for machine learning (ML) applications running on edge servers. Data transmissions from devices to servers are often over local wireless networks whose bandwidth is not just limited but, more importantly, variable. Furthermore, in cyber-physical systems interacting with the physical environment, image offloading is also commonly subject to timing constraints. It is, therefore, important to develop an adaptive approach that maximizes the inference performance of ML applications under timing constraints and the resource constraints of IoT devices. In this paper, we use image classification as our target application and propose progressive neural compression (PNC) as an efficient solution to this problem. Although neural compression has been used to compress images for different ML applications, existing solutions often produce fixed-size outputs that are unsuitable for timing-constrained offloading over variable bandwidth. To address this limitation, we train a multi-objective rateless autoencoder that optimizes for multiple compression rates via stochastic taildrop to create a compression solution that produces features ordered according to their importance to inference performance. Features are then transmitted in that order based on available bandwidth, with classification ultimately performed using the (sub)set of features received by the deadline. We demonstrate the benefits of PNC over state-of-the-art neural compression approaches and traditional compression methods on a testbed comprising an IoT device and an edge server connected over a wireless network with varying bandwidth.},
keywords = {performance evaluation;image coding;image edge detection;bandwidth;timing;internet of things;servers},
doi = {10.1109/RTSS59052.2023.00020},
url = {https://doi.ieeecomputersociety.org/10.1109/RTSS59052.2023.00020},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {dec}
}
```

