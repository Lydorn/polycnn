# Intro

Code of the IGARSS 2018 paper "End-to-End Learning of Polygons for Remote Sensing Image Classification" by Nicolas Girard and Yuliya Tarabalka.

# Dependencies

- Tensorflow 1.4

# Steps to reproduce the results of the paper

1. Train the polygon Encoder-Decoder network. This is used to pre-train the weights of the Decoder part of PolyCNN. See the corresponding [subdirectory](code/polygon_encoder_decoder).
2. Download and setup the "Distributed Solar Photovoltaic Array Location and Extent Data Set for Remote Sensing Object Identification" dataset, see the corresponding [subdirectory](data/photovoltaic_array_location_dataset).
3. Download the pre-trained InceptionV4 checkpoint, see the corresponding [subdirectory](models/inception).
4. Train PolyCNN and run inference on the test set, see the corresponding [subdirectory](code/polycnn).
5. Train the U-Net of unet_and_vectorization and run inference on the test set, see the corresponding  [subdirectory](code/unet_and_vectorization).
5. Compare the two methods, see the corresponding [subdirectory](code/evaluation).