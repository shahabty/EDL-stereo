## An unofficial implementation of "Efficient Deep Learning for Stereo Matching"

## Getting Started
This implementation is based on Tensorflow 1.12 + tf.keras and Eager Execution

### Prerequisites
Tensorflow 1.12
,Pillow
### Installing
1. Create a Anaconda3 environment with pip
2. Activate the environment
3. [install tensorflow 1.12](https://www.tensorflow.org/install)
4. Install pillow by running the following command
```
conda install -c anaconda pillow 
```
5. Go to [this link](https://drive.google.com/open?id=1pVXl-E4b5P3UsJHAb1PCFGW9EaNbq5Lh) and download "saved\_data".
6. Dump it to the project directory.
## Training and Validating
1. Training losses and validating errors are saved in "saved\_data/runs". Please follow the following instruction to see them in tensorboard.
```
cd saved_dir
tensorboard --logdir="runs" --port=6006
```
Then, open your browser and go to the link below.
```
http://computer_name_or_ip_address:6006
```
2. Qualitative examples are stored in "saved\_data/qualitative".
3. Checkpoints are stored in "saved_data/checkpoints"

## Acknowledgments
Thanks to all other people who published their implementation of the paper.

