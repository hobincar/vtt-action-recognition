import os
import json

import tensorflow as tf
import numpy as np

from config import TrainConfig as C


def get_frame_fpath(episode, frame):
    return os.path.join("./data/friends_trimmed/frames", episode, "{:05d}.jpg".format(frame)) 


def load_data(list_fpath):
    with open(list_fpath, 'r') as fin:
        data = fin.readlines()

    fpath_list = []
    label_list = []
    frame_list = []
    for d in data:
        episode, start_frame, end_frame, labels = d.split('\t')
        start_frame = int(start_frame)
        end_frame = int(end_frame)
        fpaths = [ get_frame_fpath(episode, frame) for frame in range(start_frame, end_frame+1) ]

        multi_hot = np.zeros(C.n_actions)
        if len(labels.strip()) > 0:
            labels = [ int(l) for l in labels.split(",") ]
            for label in labels:
                multi_hot[label] = 1

        target_frame = (start_frame + end_frame) // 2

        fpath_list.append(fpaths)
        label_list.append(multi_hot)
        frame_list.append(target_frame)

    fpath_list = np.asarray(fpath_list)
    label_list = np.asarray(label_list)
    frame_list = np.asarray(frame_list)
    return fpath_list, label_list, frame_list


def _train_parse_function(fpaths, label):
    def __parse_image(fpath):
        image_string = tf.read_file(fpath)
        image_decoded = tf.image.decode_jpeg(image_string)
        return image_decoded

    clip = tf.map_fn(__parse_image, fpaths, dtype=tf.uint8)
    clip = tf.stack(clip)
    clip_resized = tf.image.resize_images(clip, C.resize_shape)
    clip_random_cropped = tf.random_crop(clip_resized, [C.n_frames_per_clip, C.crop_size, C.crop_size, C.n_channels])

    return clip_random_cropped, label


def _test_parse_function(fpaths, label, frame):
    def __parse_image(fpath):
        image_string = tf.read_file(fpath)
        image_decoded = tf.image.decode_jpeg(image_string)
        return image_decoded

    clip = tf.map_fn(__parse_image, fpaths, dtype=tf.uint8)
    clip = tf.stack(clip)
    clip_resized = tf.image.resize_images(clip, C.resize_shape)
    clip_center_cropped = tf.image.resize_image_with_crop_or_pad(clip_resized, C.crop_size, C.crop_size)

    return clip_center_cropped, label, frame


def load_train_dataset(list_fpath, batch_size):
    fpaths, labels, _ = load_data(list_fpath)
    fpaths_list = tf.constant(fpaths)
    label_list = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices(( fpaths_list, label_list ))
    dataset = dataset.shuffle(buffer_size=len(fpaths))
    dataset = dataset.repeat()
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=_train_parse_function,
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=C.n_workers,
    ))
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset


def load_test_dataset(list_fpath, batch_size, shuffle=False, repeat=False):
    fpaths, labels, frames = load_data(list_fpath)
    fpaths_list = tf.constant(fpaths)
    label_list = tf.constant(labels)
    frame_list = tf.constant(frames)

    dataset = tf.data.Dataset.from_tensor_slices(( fpaths_list, label_list, frame_list ))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(fpaths))
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=_test_parse_function,
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=C.n_workers,
    ))
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset

