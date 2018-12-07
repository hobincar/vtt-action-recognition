from collections import defaultdict
import json
import os
import time

import numpy as np


def weight_classes(list_fpath, n_actions):
    if not os.path.exists(list_fpath): return None

    with open(list_fpath, 'r') as fin:
        data = fin.readlines()

    label_counter = defaultdict(lambda: 0)
    for d in data:
        _, _, _, labels, _ = d.split('\t')
        labels = [ int(l) for l in labels.split(",") ]
        for label in labels:
            label_counter[label] += 1

    weights = [ None for _ in range(n_actions) ]
    for class_idx, n_data in label_counter.items():
        weights[class_idx] = 1 / n_data
    weights = np.asarray(weights)
    weights = weights / np.sum(weights)
    return weights


class CommonConfig:
    data_root_dpath = "data"
    friends_root_dpath = os.path.join(data_root_dpath, "friends_trimmed")
    frame_root_dpath = os.path.join(friends_root_dpath, "frames")
    model_root_dpath = "models"
    output_root_dpath = "outputs"
    prediction_root_dpath = os.path.join(output_root_dpath, "predictions")
    demo_root_dpath = os.path.join(output_root_dpath, "demos")

    list_dpath = os.path.join(data_root_dpath, "list")
    annotation_dpath = os.path.join(friends_root_dpath, "annotations")
    integration_dpath = os.path.join(output_root_dpath, "integration", "data", "friends")

    frame_fpath_tpl = os.path.join(frame_root_dpath, "S{:02d}_EP{:02d}/{:05d}.jpg")
    annotation_fpath_tpl = os.path.join(annotation_dpath, "S{:02d}_EP{:02d}.json")
    list_fpath_tpl = os.path.join(list_dpath, "friends_S{:02d}_EP{:02d}.list")
    integration_fpath_tpl = os.path.join(integration_dpath, "friends_s{:02d}_e{:02d}.jsonl")

    train_list_fpath = os.path.join(list_dpath, "friends_train.list")
    test_list_fpath = os.path.join(list_dpath, "friends_test.list")

    with open("data/act2idx.json") as fin:
        act2idx = json.load(fin)
    with open('data/idx2rep.json', 'r') as fin:
        idx2rep = json.load(fin)
    with open('data/rep2idx.json', 'r') as fin:
        rep2idx = json.load(fin)
    with open("data/rep2sta.json") as fin:
        rep2sta = json.load(fin)
    n_actions = len(idx2rep)
    actions = list(idx2rep.values())
    action_labels = list(idx2rep.keys())

    fps_used_to_extract_frames = 5
    n_frames_per_clip = 16

    model_tag = "C3D"

    full_shape = { "width": 1280, "height": 720 } # [ width, height ]
    resize_shape = { "width": 171, "height": 128 } # [ height, width ]
    n_channels = 3


class ListConfig(CommonConfig):
    seasons = [ 1 ]
    episodes_list = [ range(0, 24) ]

    train_ratio = 0.7

    bbox_tag = "full_rect" # [ "face_rect", "full_rect" ]
    bbox_labels = [ "min_x", "min_y", "max_x", "max_y" ]

    n_front = CommonConfig.n_frames_per_clip // 2 - 1
    n_back = CommonConfig.n_frames_per_clip - n_front - 1


class DataLoaderConfig(CommonConfig):
    use_bbox = True
    bbox_mode = "fit" # [ "fit", "center_pad" ]

    crop_size = 112

    batch_size = 30


class TrainConfig(DataLoaderConfig):
    n_workers = 4

    use_pretrained_model = False
    if use_pretrained_model:
        pretrained_model_dpath = "pretrained_models"
        pretrained_model_name = "sports1m_finetuning_ucf101"
        pretrained_model_fpath = os.path.join(pretrained_model_dpath, "{}.model".format(pretrained_model_name))

    n_iterations = 50000
    train_log_every = 100
    test_log_every = 1000
    high_prob_threshold = 0.5
    n_log_every = 5
    log_topk = 5
    save_every = 10000
    moving_average_decay = 0.9999
    lr_stable = 1e-3
    lr_finetune = 1e-3

    class_weights = weight_classes(DataLoaderConfig.train_list_fpath, DataLoaderConfig.n_actions)

    timestamp = time.strftime("%y%m%d-%H:%M:%S", time.gmtime())
    id = "{} | bbox-{} | lr-st-{}-fn-{} | pt-{} | {}".format(
        DataLoaderConfig.model_tag, 'ON' if DataLoaderConfig.use_bbox else 'OFF', lr_stable, lr_finetune,
        pretrained_model_name if use_pretrained_model else "None", timestamp)

    log_root_dpath = "logs"
    log_dpath = os.path.join(log_root_dpath, id)

    model_fpath = os.path.join(DataLoaderConfig.model_root_dpath, id, "model")


class PredConfig(DataLoaderConfig):
    seasons = [ 1 ]
    episodes_list = [ range(1, 24) ]
    model_name = "C3D | bbox-ON | lr-st-1e-05-fn-0.001 | pt-None | 181206-19:35:34"
    n_iterations = 30000

    topk = 5

    model_fpath = os.path.join(DataLoaderConfig.model_root_dpath, model_name, "model-{}".format(n_iterations))
    prediction_fpath_tpl = os.path.join(DataLoaderConfig.prediction_root_dpath, model_name, str(n_iterations),
                                        "S{:02d}_EP{:02d}.json")


class DemoConfig(PredConfig):
    demo_fpath_tpl = os.path.join(PredConfig.demo_root_dpath, PredConfig.model_name, str(PredConfig.n_iterations),
                                  "S{:02d}_EP{:02d}.mp4")

