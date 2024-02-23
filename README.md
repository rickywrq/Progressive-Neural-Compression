# Progressive-Neural-Compression

### [Feb, 2024] We are actively working on this repo. If you have any questions, please contact the first author. The network and the checkpoints are in the `demo_simulation` folder. To train PNC from scratch, please go to `training`.

## Introduction

This repository contains the source code and testbed setup instructions for **R. Wang, H. Liu, J. Qiu, M. Xu, R. Guerin, C. Lu, [Progressive Neural Compression for Adaptive Image Offloading under Timing Constraints](https://doi.org/10.1109/RTSS59052.2023.00020), IEEE Real-Time Systems Symposium ([RTSS'23](https://2023.rtss.org/)), December 2023.  [[IEEE Xplore](https://doi.org/10.1109/RTSS59052.2023.00020)]** [[arXiv](https://arxiv.org/pdf/2310.05306.pdf)] [[RTSS 2023 Proceedings](https://doi.ieeecomputersociety.org/10.1109/RTSS59052.2023.00020)]

🏆🤫 *This paper won "Outstanding Paper Award" and "[RTSS 2023 Best Student Paper Award](https://2023.rtss.org/awards/)."* 🌟🥇 Great thanks to everyone on the team.

![pnc_overview](assets/pnc_overview.png)


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
@INPROCEEDINGS{10405983,
  author={Wang, Ruiqi and Liu, Hanyang and Qiu, Jiaming and Xu, Moran and Guérin, Roch and Lu, Chenyang},
  booktitle={2023 IEEE Real-Time Systems Symposium (RTSS)}, 
  title={Progressive Neural Compression for Adaptive Image Offloading Under Timing Constraints}, 
  year={2023},
  volume={},
  number={},
  pages={118-130},
  keywords={Performance evaluation;Image coding;Image edge detection;Bandwidth;Timing;Internet of Things;Servers;neural compression;edge offloading;image classification;real-time transmission},
  doi={10.1109/RTSS59052.2023.00020}}

```

