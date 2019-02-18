from os.path import join
import tensorflow as tf
import random
import pickle
from model import SiameseStereoMatching
from dataset import Dataset
tf.enable_eager_execution() 
#tf.keras.backend.set_session(tf_debug.LocalCLIDebugWrapperSession(tf.Session()))

args = {
'data_path':'/mnt/creeper/grad/nabaviss/kitti/training',
'out_cache_path':'saved_data/locs',
'exp_dir': 'saved_data',
'left_img_folder': 'image_2',  #RGB Left
'right_img_folder': 'image_3', #RGB Right
'disparity_folder': 'disp_noc_0',
'num_input_channels': 3,
'img_height': 370,
'img_width': 1224,
'batch_size': 128,
'patch_size':37,
'val_freq':1000,
'disparity_range':201,
'half_patch_size':37//2,
'half_range': 201//2,
'num_train': 160,
'device': '/gpu:0',
'learning_rate':0.01,
'num_iterations':40000,
'seed':3,

}

def main(args):
    random.seed(args['seed'])
    global_step = tf.get_variable('global_step',initializer = 0,trainable = False)

    with tf.device(args['device']):
        model = SiameseStereoMatching(args,global_step) 
    boundaries, lr_values = [24000, 32000], [args['learning_rate'],args['learning_rate']/5,args['learning_rate']/25]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, lr_values)
    optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)

    patch_locations_path = join(args['out_cache_path'], 'patch_locations.pkl')
    #loading patches
    with open(patch_locations_path, 'rb') as handle:
        patch_locations = pickle.load(handle)

    training_dataset = Dataset(args, patch_locations, phase='train')
    validation_dataset = Dataset(args, patch_locations, phase='val')

    model.run(training_dataset, validation_dataset, optimizer,args,tensorboard = True)

if __name__ == "__main__":
    main(args)


