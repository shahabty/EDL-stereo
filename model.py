import os
from os.path import join
import numpy as np
from collections import namedtuple
from PIL import Image

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
import tensorflow.keras.backend as K
from tqdm import tqdm

Batch = namedtuple('Batch', ['left', 'right', 'label'])

class Writer():

    def __init__(self,args,path = 'runs'):
        self.writer = tf.contrib.summary.create_file_writer(path, flush_millis=10000)
        self.writer.set_as_default()
        self.args = args

    def log_to_tensorboard(self,name,tensor_to_print = None,step = None):
        step = tf.cast(step,tf.int64)
        if name == 'train_loss':
            tf.contrib.summary.scalar(name,tensor_to_print,step = step)
            tf.contrib.summary.scalar('global_step', step)
            tf.contrib.summary.flush()
        if name == 'error':
            tf.contrib.summary.scalar(name+'_2',tensor_to_print[0]/tensor_to_print[4],step = step)
            tf.contrib.summary.flush()
            tf.contrib.summary.scalar(name+'_3',tensor_to_print[1]/tensor_to_print[4],step = step)
            tf.contrib.summary.flush()
            tf.contrib.summary.scalar(name+'_4',tensor_to_print[2]/tensor_to_print[4],step = step)
            tf.contrib.summary.flush()
            tf.contrib.summary.scalar(name+'_5',tensor_to_print[3]/tensor_to_print[4],step = step)
            tf.contrib.summary.flush()

        if name == 'qualitative':
            """ There is a very recent bug in tf.contrib.summary.image. So, I saved the images in a folder instead of visualizing them in tensorboard."""
            disp_img = np.array(tensor_to_print[0])
            disp_img[disp_img < 0] = 0
            disp_img_save = disp_img*(255.0/disp_img.max())
            disp_img_save = np.repeat(disp_img_save[:, :, np.newaxis], 3, axis=2)
            left_img_save = np.array(tensor_to_print[1])[0]
            left_img_save = self._normalize_uint8(left_img_save)
            right_img_save = np.array(tensor_to_print[2])[0]
            right_img_save = self._normalize_uint8(right_img_save)

            disp_img_pil = Image.fromarray(disp_img_save.astype('uint8'),'RGB').save(self.args['exp_dir']+"/"+name+"/disp_img_"+str(int(step.numpy()))+ ".png")
            left_img_pil = Image.fromarray(left_img_save.astype('uint8'),'RGB').save(self.args['exp_dir']+"/"+name+"/left_img_"+str(int(step.numpy()))+ ".png")
            right_img_pil = Image.fromarray(right_img_save.astype('uint8'),'RGB').save(self.args['exp_dir']+"/"+name+"/right_img_"+str(int(step.numpy()))+ ".png")
            
    def _normalize_uint8(self, array):
        array = (array - array.min()) / (array.max() - array.min())
        array = (array * 255).astype(np.uint8)
        return array

class SiameseStereoMatching(tf.keras.Model):
    def __init__(self, args, global_step):
        """Constructor."""
        super(SiameseStereoMatching, self).__init__()
        self._device = args['device']
        self._exp_dir = args['exp_dir']
        self._num_input_channels = args['num_input_channels']
        self._global_step = global_step
        self._disparity_range = args['disparity_range']
        self._half_range = args['half_range']
        self.patch_feature_module = self._create_patch_feature_module(self._num_input_channels)

    def _create_patch_feature_module(self, num_input_channels):
        c = num_input_channels
        patch_feature_module = tf.keras.Sequential()

        patch_feature_module.add(self._create_conv_bn_relu_module(c, 32, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(32, 32, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(32, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5))
        patch_feature_module.add(self._create_conv_bn_relu_module(64, 64, 5, 5,add_relu=False))
        return patch_feature_module

    def _create_conv_bn_relu_module(self, num_input_channels, num_output_channels,
                                    kernel_height, kernel_width, add_relu=True):
        conv_bn_relu = tf.keras.Sequential()
        conv = tf.keras.layers.Conv2D(num_input_channels,
                                      (kernel_height, kernel_width),
                                      padding='valid',
                                      kernel_initializer=tf.keras.initializers.he_uniform())
        bn = tf.keras.layers.BatchNormalization()
        relu = tf.keras.layers.ReLU()

        conv_bn_relu.add(conv)
        conv_bn_relu.add(bn)
        if add_relu:
            conv_bn_relu.add(relu)

        return conv_bn_relu

    def compute(self, batch, training=None):
        if training == True:
            with tfe.GradientTape() as tape:
                inner_product = self.call(batch.left, batch.right,training)
                loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(labels=batch.label,logits=inner_product))
                return tape.gradient(loss, self.variables), loss
        else:
            cost_volume,win_indices= self.call(batch.left, batch.right,training)
            inner_product = cost_volume
            img_height, img_width = cost_volume.shape[1], cost_volume.shape[2] # 1 x H x W x C
            cost_volume = tf.pad(cost_volume, tf.constant([[0, 0,], [2, 2,], [2, 2], [0, 0]]),"REFLECT")
            cost_volume = tf.layers.average_pooling2d(cost_volume,pool_size=(5, 5),strides=(1, 1),padding='VALID',data_format='channels_last')
            cost_volume = tf.squeeze(cost_volume)
            row_indices, _ = tf.dtypes.cast(tf.meshgrid(tf.range(0, img_width),tf.range(0, img_height)),dtype=tf.int64)
            disp_prediction_indices = tf.argmax(cost_volume, axis=-1)
            disp_prediction = []
            for i in range(img_width):
                column_disp_prediction = tf.gather(win_indices[i],disp_prediction_indices[:, i],axis=0)
                column_disp_prediction = tf.expand_dims(column_disp_prediction, 1)
                disp_prediction.append(column_disp_prediction)

            disp_prediction = tf.concat(disp_prediction, 1)
            disp_prediction = row_indices - disp_prediction

            return disp_prediction

    def run(self, training_dataset, validation_dataset, optimizer,args,tensorboard = False):
        #The only option :) tensorboard = True
        if tensorboard == True:
            writer = Writer(args,path = args['exp_dir']+'/runs')
        else:
            print('Please use tensorboard!')
            return 0

        with tf.contrib.summary.record_summaries_every_n_global_steps(1):
            with tf.device(self._device):
                for itx in tqdm(range(0,args['num_iterations']//args['val_freq']),desc = 'iterations:'):
                    try:
                        print("training...")
                        # Training iterations.
                        train_loss = 0.0
                        for i in range(args['val_freq']):
                            left_patches, right_patches, labels = training_dataset.iterator.get_next()
                            batch = Batch(left_patches, right_patches, labels)
                            grads, t_loss = self.compute(batch, training=True)
                            optimizer.apply_gradients(zip(grads, self.variables))
                            train_loss += t_loss

                        writer.log_to_tensorboard('train_loss',train_loss/args['val_freq'],step = self._global_step)

                        print('validating...')
                        # Validation iterations.
                        error_pixel_2 = 0.0
                        error_pixel_3 = 0.0
                        error_pixel_4 = 0.0
                        error_pixel_5 = 0.0
                        for i,idx in enumerate(validation_dataset.sample_ids):
                            left_image,right_image,disparity_ground_truth = validation_dataset.left_images[idx],validation_dataset.right_images[idx],validation_dataset.disparity_images[idx]
                            paddings = validation_dataset.get_paddings()
                            left_image = tf.pad(tf.expand_dims(left_image, 0),paddings, "CONSTANT")
                            right_image = tf.pad(tf.expand_dims(right_image, 0),paddings, "CONSTANT")
                            batch = Batch(left_image,right_image,disparity_ground_truth)
                            disparity_prediction = self.compute(batch, training = False)

                            valid_gt_pixels = (disparity_ground_truth != 0).astype('float')
                            masked_prediction_valid = disparity_prediction * valid_gt_pixels
                            num_valid_gt_pixels = valid_gt_pixels.sum()

                            error_pixel_2 += (np.abs(masked_prediction_valid -disparity_ground_truth) > 2).sum() / num_valid_gt_pixels
                            error_pixel_3 += (np.abs(masked_prediction_valid -disparity_ground_truth) > 3).sum() / num_valid_gt_pixels
                            error_pixel_4 += (np.abs(masked_prediction_valid -disparity_ground_truth) > 4).sum() / num_valid_gt_pixels
                            error_pixel_5 += (np.abs(masked_prediction_valid -disparity_ground_truth) > 5).sum() / num_valid_gt_pixels

                        #print('----------validation summary--------------')
                        #print(error_pixel_2 / len(validation_dataset.sample_ids))
                        #print(error_pixel_3 / len(validation_dataset.sample_ids))
                        #print(error_pixel_4 / len(validation_dataset.sample_ids))
                        #print(error_pixel_5 / len(validation_dataset.sample_ids))

                        writer.log_to_tensorboard('error',(error_pixel_2,error_pixel_3,error_pixel_4,error_pixel_5,len(validation_dataset.sample_ids)),step = self._global_step)


                        print('Saving one random prediciton...')
                        random_img_idx = np.random.choice(validation_dataset.sample_ids)
                        sample_left_image,sample_right_image,disparity_prediction = validation_dataset.left_images[random_img_idx],validation_dataset.right_images[random_img_idx],validation_dataset.disparity_images[random_img_idx]
                        paddings = validation_dataset.get_paddings()
                        sample_left_image = tf.pad(tf.expand_dims(sample_left_image, 0),paddings, "CONSTANT")
                        sample_right_image = tf.pad(tf.expand_dims(sample_right_image, 0),paddings, "CONSTANT")
                        batch = Batch(sample_left_image,sample_right_image,disparity_prediction)
                        disparity_prediction = self.compute(batch,training = False)
                        writer.log_to_tensorboard('qualitative',(disparity_prediction,sample_left_image,sample_right_image),step = self._global_step)
                        self._global_step.assign_add(args['val_freq'])

                        # Save checkpoint.
                        tfe.Saver(self.variables).save(join(self._exp_dir, 'checkpoints', 'checkpoints'),global_step=self._global_step)
                    except tf.errors.OutOfRangeError:
                        break

    def call(self, left_input, right_input, training=None, mask=None):
        left_feature = self.patch_feature_module(left_input, training=training)
        right_feature = self.patch_feature_module(right_input, training=training)

        if training == False:
            inner_product, win_indices = [], []
            img_height, img_width = right_feature.shape[1], right_feature.shape[2]
            row_indices = tf.dtypes.cast(tf.range(0, img_width), dtype=tf.int64)

            for i in range(img_width):
                left_column_features = tf.squeeze(left_feature[:, :, i])
                start_win = max(0, i - self._half_range)
                end_win = max(self._disparity_range, self._half_range + i + 1)
                start_win = start_win - max(0, end_win - img_width.value)
                end_win = min(img_width, end_win)

                right_win_features = tf.squeeze(right_feature[:, :, start_win:end_win])
                win_indices_column = tf.expand_dims(row_indices[start_win:end_win], 0)
                inner_product_column = tf.einsum('ij,ikj->ik', left_column_features,
                                                 right_win_features)
                inner_product_column = tf.expand_dims(inner_product_column, 1)
                inner_product.append(inner_product_column)
                win_indices.append(win_indices_column)

            inner_product = tf.expand_dims(tf.concat(inner_product, 1), 0)
            win_indices = tf.concat(win_indices, 0)

            return inner_product, win_indices
        else:
            left_feature = tf.squeeze(left_feature)
            inner_product = tf.einsum('il,ijkl->ijk', left_feature, right_feature)

            return inner_product
