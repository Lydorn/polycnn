# Scripts to execute

Various scripts have to be executed in sequence, they are numbered from 0 to 4.

## 0 Prepare Dataset

The script 0_prepare_dataset.py:
 - Reads polygons and images
 - Converts non-RGB images to RGB (only one image is actually non-RGB)
 - Simplifies polygons because some polygons are over-defined (currently using a tolerance of 1 pixel for the Douglas-Peucker algorithm)
 - Outputs a .tfrecord file

Some parameters at the beginning of the script can be changed.

## 1 Filter Dataset

The script 1_filter_dataset removes images that are too big.

Some parameters at the beginning of the script can be changed.

## 2 Split Dataset

The script 2_split_dataet.py creates train, val and test datasets from the filtered dataset.

Some parameters at the beginning of the script can be changed.

## 3 Train

The 3_train.py script performs the training of the model on the train dataset and validates on the val dataset.

Visualize training using TensorBoard (you might have to modify the --logdir argument according to your directory structure):
```
python -m tensorboard.main --logdir=~/polycnn/code/polycnn/runs
```

Some parameters at the beginning of the script can be changed.

This takes several hours to train for 100000 iterations with a batch size of 128. Although good results are already obtained at 5000 iterations (approx. 1 hour of training).

## 4 Inference

Compute polygon predictions on the test set using the trained model and saves the results (used later on to measure accuracy performance)

# Scripts used by the previou sones

## Dataset

The dataset.py script is used by the training algorithm to fetch examples from a tfrecord file.
It crops or pads images to a given constant resolution.

Some parameters at the beginning of the script can be changed.

## Loss

The loss.py script defines the different loss functions used for training.

## Model

The model.py script defines the graph of the model.