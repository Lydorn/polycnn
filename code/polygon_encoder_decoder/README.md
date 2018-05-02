## 0 Generate Dataset

The 0_generated_dataset.py script generates the synthetic polygon dataset. Execute once before training.

Some parameters at the beginning of the script can be changed.

## 1 Train

The 1_train.py is to be executed to train the model.

Visualize training using TensorBoard (you might have to modify the --logdir argument according to your directory structure):
```
python -m tensorboard.main --logdir=~/polycnn/projects/polygon/encoder_decoder/runs
```

Some parameters at the beginning of the script can be changed.
