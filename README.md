# PolyCNN

This is the official code for the paper:

**End-to-End Learning of Polygons for Remote Sensing Image Classification**\
[Nicolas Girard](https://www-sop.inria.fr/members/Nicolas.Girard/),
[Yuliya Tarabalka](https://www-sop.inria.fr/members/Yuliya.Tarabalka/)\
IGARSS 2018

**[[Paper](https://www-sop.inria.fr/members/Nicolas.Girard/pdf/Girard_2018_IGARSS_paper.pdf)] [[Slides](https://www-sop.inria.fr/members/Nicolas.Girard/pdf/Girard_2018_IGARSS_slides.pdf)]**

# Dependencies

- Tensorflow 1.4

# Steps to reproduce the results of the paper

1. Train the polygon Encoder-Decoder network. This is used to pre-train the weights of the Decoder part of PolyCNN. See the corresponding [subdirectory](code/polygon_encoder_decoder).
2. Download and setup the "Distributed Solar Photovoltaic Array Location and Extent Data Set for Remote Sensing Object Identification" dataset, see the corresponding [subdirectory](data/photovoltaic_array_location_dataset).
3. Download the pre-trained InceptionV4 checkpoint, see the corresponding [subdirectory](models/inception).
4. Train PolyCNN and run inference on the test set, see the corresponding [subdirectory](code/polycnn).
5. Train the U-Net of unet_and_vectorization and run inference on the test set, see the corresponding  [subdirectory](code/unet_and_vectorization).
5. Compare the two methods, see the corresponding [subdirectory](code/evaluation).

### If you use this code for your own research, please cite:

```
@inproceedings{girard:hal-01762446,
  TITLE = {{End-to-End Learning of Polygons for Remote Sensing Image Classification}},
  AUTHOR = {Girard, Nicolas and Tarabalka, Yuliya},
  URL = {https://hal.inria.fr/hal-01762446},
  BOOKTITLE = {{IEEE International Geoscience and Remote Sensing Symposium -- IGARSS 2018}},
  ADDRESS = {Valencia, Spain},
  YEAR = {2018},
  MONTH = Jul,
  KEYWORDS = {convolutional neural networks ; Index Terms- High-resolution aerial images ;  polygon ; vectorial ;  regression ;  deep learning},
  PDF = {https://hal.inria.fr/hal-01762446/file/girard.pdf},
  HAL_ID = {hal-01762446},
  HAL_VERSION = {v1},
}
```