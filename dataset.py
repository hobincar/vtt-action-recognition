from collections import defaultdict
import os
import json

import tensorflow as tf
import numpy as np

from config import TrainConfig as C


def get_frame_fpath(episode, frame):
    return os.path.join(C.frame_root_dpath, episode, "{:05d}.jpg".format(frame)) 


def __merge(dups):
    episode_id, start_frame, end_frame, _, _ = dups[0]
    labels = []
    for dup_episode_id, dup_start_frame, dup_end_frame, dup_labels, _ in dups:
        assert dup_episode_id == episode_id
        assert dup_start_frame == start_frame
        assert dup_end_frame == end_frame
        labels += dup_labels
    labels = list(set(labels))
    bbox = [ "0", "0", str(C.full_shape["width"]), str(C.full_shape["height"]) ]

    labels = ','.join(labels)
    bbox = ','.join(bbox)
    return ( episode_id, start_frame, end_frame, labels, bbox )


def merge_data_along_frame(data):
    merge_dict = defaultdict(lambda: [])
    for d in data:
        episode, start_frame, _, _, _ = d
        id = "_".join([ episode, start_frame ])
        merge_dict[id].append(d)

    merged_data = list(map(__merge, merge_dict.values()))
    return merged_data


def load_data(list_fpath):
    with open(list_fpath, 'r') as fin:
        data = fin.readlines()
    data = [ d.split('\t') for d in data ]
    data = [ [ e.strip() for e in d ] for d in data ]

    if not C.use_bbox:
        data = merge_data_along_frame(data)

    fpaths_list = []
    labels_list = []
    bbox_list = []
    frame_list = []
    for episode, start_frame, end_frame, labels, bbox in data:
        start_frame = int(start_frame)
        end_frame = int(end_frame)

        # fpaths
        fpaths = [ get_frame_fpath(episode, frame) for frame in range(start_frame, end_frame+1) ]

        # labels
        multi_hot = np.zeros(C.n_actions)
        if len(labels) > 0:
            labels = [ int(l) for l in labels.split(",") ]
            for label in labels:
                multi_hot[label] = 1

        # bbox
        bbox = bbox.split(',')
        bbox = [ int(c) for c in bbox ]

        # frame
        target_frame = (start_frame + end_frame) // 2

        fpaths_list.append(fpaths)
        labels_list.append(multi_hot)
        bbox_list.append(bbox)
        frame_list.append(target_frame)

    fpaths_list = np.asarray(fpaths_list)
    labels_list = np.asarray(labels_list)
    bbox_list = np.asarray(bbox_list)
    frame_list = np.asarray(frame_list)
    return fpaths_list, labels_list, bbox_list, frame_list


def _parse_function(fpaths, label, bbox, frame):
    def __parse_image(fpath):
        image_string = tf.read_file(fpath)
        image_decoded = tf.image.decode_jpeg(image_string)
        return image_decoded

    clip = tf.map_fn(__parse_image, fpaths, dtype=tf.uint8)
    clip = tf.stack(clip)


    bbox_dict_group = {}
    if C.use_bbox:
        bbox_dict_group["original"] = {
            "min_x": bbox[0],
            "min_y": bbox[1],
            "max_x": bbox[2],
            "max_y": bbox[3],
        }
        resize_ratio = {
            "width": tf.constant(C.full_shape["width"] / C.resize_shape["width"], dtype=tf.float32),
            "height": tf.constant(C.full_shape["height"] / C.resize_shape["height"], dtype=tf.float32),
        }

        if C.bbox_mode == "fit":
            x1 = tf.cast(bbox[0], tf.int32)
            y1 = tf.cast(bbox[1], tf.int32)
            w = tf.cast(bbox[2] - bbox[0], tf.int32)
            h = tf.cast(bbox[3] - bbox[1], tf.int32)
            clip_cropped = tf.image.crop_to_bounding_box(clip, y1, x1, h, w)
            clip_cropped = tf.image.resize_images(clip_cropped, ( C.crop_size, C.crop_size ))

            bbox_dict_group["resized"] = {
                "min_x": tf.cast(tf.cast(bbox[0], tf.float32) * resize_ratio["width"], tf.int32),
                "min_y": tf.cast(tf.cast(bbox[1], tf.float32) * resize_ratio["height"], tf.int32),
                "max_x": tf.cast(tf.cast(bbox[2], tf.float32) * resize_ratio["width"], tf.int32),
                "max_y": tf.cast(tf.cast(bbox[3], tf.float32) * resize_ratio["height"], tf.int32),
            }
            bbox_dict_group["resize2original"] = {
                "min_x": bbox[0],
                "min_y": bbox[1],
                "max_x": bbox[2],
                "max_y": bbox[3],
            }
        elif C.bbox_mode == "center_pad":
            center_x = (bbox[2] - bbox[0]) / 2
            center_y = (bbox[3] - bbox[1]) / 2

            center_x_resized = tf.cast(center_x, tf.float32) * resize_ratio["width"]
            center_x_resized = tf.math.minimum(center_x_resized,  tf.cast(C.resize_shape["width"] - C.crop_size / 2, tf.float32))
            center_x_resized = tf.math.maximum(center_x_resized, tf.cast(C.crop_size / 2, tf.float32))
            center_y_resized = tf.cast(center_y, tf.float32) * resize_ratio["height"]
            center_y_resized = tf.math.minimum(center_y_resized,  tf.cast(C.resize_shape["height"] - C.crop_size / 2, tf.float32))
            center_y_resized = tf.math.maximum(center_y_resized, tf.cast(C.crop_size / 2, tf.float32))

            x1 = center_x_resized - tf.cast(C.crop_size / 2, tf.float32)
            x1 = tf.cast(x1, tf.int32)
            y1 = center_y_resized - tf.cast(C.crop_size / 2, tf.float32)
            y1 = tf.cast(y1, tf.int32)

            clip_resized = tf.image.resize_images(clip, ( C.resize_shape["height"], C.resize_shape["width"] ))
            clip_cropped = tf.image.crop_to_bounding_box(clip_resized, x1, y1, C.crop_size, C.crop_size)
            clip_cropped = tf.image.resize_images(clip_cropped, ( C.crop_size, C.crop_size ))
        else:
            raise NotImplementedError("Unknown bbox_mode: {}".format(C.bbox_mode))
    else:
        clip_resized = tf.image.resize_images(clip, ( C.resize_shape["height"], C.resize_shape["width"] ))
        clip_cropped = tf.image.resize_image_with_crop_or_pad(clip_resized, C.crop_size, C.crop_size)

    return clip_cropped, label, bbox_dict_group, frame


def load_dataset(list_fpath, batch_size, shuffle=False, repeat=False):
    fpaths_list, labels_list, bbox_list, frame_list = load_data(list_fpath)

    dataset = tf.data.Dataset.from_tensor_slices(( fpaths_list, labels_list, bbox_list, frame_list ))
    if shuffle:
        n_data = len(fpaths_list)
        dataset = dataset.shuffle(buffer_size=n_data)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.apply(tf.data.experimental.map_and_batch(
        map_func=_parse_function,
        batch_size=batch_size,
        drop_remainder=True,
        num_parallel_calls=C.n_workers,
    ))
    dataset = dataset.prefetch(buffer_size=batch_size)
    return dataset

