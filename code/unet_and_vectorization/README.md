# Scripts to execute

Various scripts have to be executed in sequence, they are numbered from 0 to 4.

## 0 Adapt Dataset

Adds a raster of the polygon for each sample of the tfrecords for U-Net training.

## 1 Train

The 1_train.py script performs the training of the model on the train dataset and validates on the val dataset.

Visualize training using TensorBoard (you might have to modify the --logdir argument according to your directory structure):
```
python -m tensorboard.main --logdir=~/polycnn/code/unet_and_vectorization/runs
```

Some parameters at the beginning of the script can be changed.

## 2 Inference

Compute polygon predictions on the test set using the trained model + vectorization and saves the results (used later on to measure accuracy performance).