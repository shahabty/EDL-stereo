# An unofficial implementation of "Efficient Deep Learning for Stereo Matching"

## Getting Started
This implementation is based on Tensorflow 1.12 + tf.keras and Eager Execution

### Prerequisites
Tensorflow 1.12
pillow
### Installing
1. Create a Anaconda3 environment with pip
2. Activate the environment
2. [install tensorflow 1.12](https://www.tensorflow.org/install)
2. Install pillow by running the following command
```
conda install -c anaconda pillow 

```

## Training and Validating
1. Training losses and validating errors are saved in "saved\_dir". Please follow the following instruction to see them in tensorboard.
```
cd saved_dir
tensorboard --logdir="runs" --port=6006

```
Then, open your browser and go to the link below.
```
http://computer_name_or_ip:6006

```
2. Qualitative examples are stored in "saved\_dir/qualitative".
3. Checkpoints are stored in "saved_dir/checkpoints"

## Acknowledgments




