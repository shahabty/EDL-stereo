import os
from os.path import join
import glob
import pickle
from PIL import Image

import numpy as np
import tensorflow as tf

import tensorflow.keras.backend as K  

def trim_image(img, img_height, img_width):
    return img[0:img_height, 0:img_width]


def load_image_paths(data_path, left_img_folder, right_img_folder,
                     disparity_folder):
    left_image_paths = sorted(glob.glob(join(data_path, left_img_folder, '*10.png')))
    right_image_paths = sorted(glob.glob(join(data_path, right_img_folder, '*10.png')))
    disparity_image_paths = sorted(glob.glob(join(data_path, disparity_folder, '*10.png')))

    return left_image_paths, right_image_paths, disparity_image_paths

def _load_image(image_path, img_height, img_width):
    image_file = tf.read_file(image_path)
    img = tf.image.decode_png(image_file)
    img = trim_image(img, img_height, img_width)
    img = tf.image.per_image_standardization(img)
    return img


def _load_disparity(image_path, img_height, img_width):
    disp_img = np.array(Image.open(image_path)).astype('float64')
    disp_img = trim_image(disp_img, img_height, img_width)
    disp_img /= 256

    return disp_img

def _load_images(left_image_paths, right_image_paths, disparity_paths, img_height, img_width):
    left_images = []
    right_images = []
    disparity_images = []
    for idx in range(left_image_paths.shape[0]):
        left_images.append(_load_image(left_image_paths[idx], img_height, img_width))
        right_images.append(_load_image(right_image_paths[idx], img_height, img_width))

        if disparity_paths:
            disparity_images.append(_load_disparity(disparity_paths[idx], img_height, img_width))

    return (tf.convert_to_tensor(left_images),
            tf.convert_to_tensor(right_images),
            np.array(disparity_images))


def _get_labels(disparity_range, half_range):
    gt = np.zeros((disparity_range))

    # NOTE: Smooth targets are [0.05, 0.2, 0.5, 0.2, 0.05], hard-coded.
    gt[half_range - 2: half_range + 3] = np.array([0.05, 0.2, 0.5, 0.2, 0.05])

    return gt


class Dataset:
    def __init__(self, args, patch_locations, phase):
        self._args = args
        left_image_paths, right_image_paths, disparity_paths = \
                load_image_paths(args['data_path'],
                                 args['left_img_folder'],
                                 args['right_img_folder'],
                                 args['disparity_folder'])

        left_image_paths = tf.constant(left_image_paths)
        right_image_paths = tf.constant(right_image_paths)
        self.left_images, self.right_images, self.disparity_images = \
                _load_images(left_image_paths,
                             right_image_paths,
                             disparity_paths,
                             args['img_height'],
                             args['img_width'])
        self.iterator = self._create_dataset_iterator(patch_locations, phase)
        if phase == 'train':
            self.sample_ids = patch_locations['train_ids']
        elif phase == 'val':
            self.sample_ids = patch_locations['val_ids']

    def get_paddings(self):
        return tf.constant([[0, 0,],
                            [self._args['half_patch_size'], self._args['half_patch_size']],
                            [self._args['half_patch_size'], self._args['half_patch_size']],
                            [0, 0]])

    def _parse_function(self, sample_info):
        """Parsing function passed to map operation for loading data."""
        idx = tf.to_int32(sample_info[0])
        left_center_x = tf.to_int32(sample_info[1])
        left_center_y = tf.to_int32(sample_info[2])
        right_center_x = tf.to_int32(sample_info[3])

        left_image = self.left_images[idx]
        right_image = self.right_images[idx]

        left_patch = left_image[left_center_y -\
                                self._args['half_patch_size']:left_center_y +\
                                self._args['half_patch_size'] + 1,
                                left_center_x -\
                                self._args['half_patch_size']:left_center_x +\
                                self._args['half_patch_size'] + 1, :]
        right_patch = right_image[left_center_y -\
                                  self._args['half_patch_size']:left_center_y +\
                                  self._args['half_patch_size'] + 1,
                                  right_center_x - self._args['half_patch_size'] -\
                                  self._args['half_range']:right_center_x +\
                                  self._args['half_patch_size'] +\
                                  self._args['half_range'] + 1, :]


        labels = tf.convert_to_tensor(_get_labels(self._args['disparity_range'],
                                                  self._args['half_range']))

        return left_patch, right_patch, labels

    def _create_train_iterator(self, patch_locations):
        """Create training data iterator."""
        dataset_locations = patch_locations['valid_locations_train']

        dataset = tf.data.Dataset.from_tensor_slices(dataset_locations)
        dataset = dataset.map(self._parse_function)
        batched_dataset = dataset.batch(self._args['batch_size'])
        iterator = batched_dataset.make_one_shot_iterator()

        return iterator

    def _create_val_iterator(self, patch_locations):
        """Create validation data iterator."""
        dataset_locations = patch_locations['valid_locations_val']
        # NOTE: Repeat dataset so that we can have 40k iterations.
        dataset_locations = dataset_locations.repeat(2, axis=0)

        dataset = tf.data.Dataset.from_tensor_slices(dataset_locations)
        dataset = dataset.map(self._parse_function)
        batched_dataset = dataset.batch(self._args['batch_size'])
        iterator = batched_dataset.make_one_shot_iterator()

        return iterator

    def _create_dataset_iterator(self, patch_locations, phase='train'):
        """Create dataset iterator."""
        if phase == 'train':
            return self._create_train_iterator(patch_locations)
        elif phase == 'val':
            return self._create_val_iterator(patch_locations)
