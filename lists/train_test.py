"""
Format:
    S01_EP01    13  17    4, 7, 8, 9    123, 234, 345, 456
    ...
"""

import os
import random
random.seed(42)

from config import ListConfig as C
from lists.utils import load_annotation, parse_frame_number, timestr_to_seconds, get_endpoints_from_median_frame, \
                        merge_duplicates
import utils


def load_annotations():
    annotations = []
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            annotation = load_annotation(season, episode)
            annotations.append(( season, episode, annotation ))
    return annotations


def parse_annotation(annotation):
    start_time = annotation["start_time"]
    start_seconds = timestr_to_seconds(start_time)
    start_frame = int(start_seconds * C.fps_used_to_extract_frames)

    end_time = annotation["end_time"]
    end_seconds = timestr_to_seconds(end_time)
    end_frame = int(end_seconds * C.fps_used_to_extract_frames)

    labels = []
    bboxes = []
    for person, info in annotation["person"][0].items():
        info = info[0]

        action = info["behavior"]
        if action in C.act2idx:
            label = C.act2idx[action]
            label = str(label)
            rect = info[C.bbox_tag]
            try:
                for bbox_label in C.bbox_labels:
                    int(rect[bbox_label])
            except:
                continue
            bbox = [ rect[bbox_label] for bbox_label in C.bbox_labels ]

            labels.append(label)
            bboxes.append(bbox)

    return start_frame, end_frame, labels, bboxes


def parse_total_list():
    total_list = []

    annotations_list = load_annotations()
    for season, episode, annotations in annotations_list:
        episode_id = utils.format_episode_id(season, episode)

        frame_fnames = os.listdir(os.path.join(C.frame_root_dpath, episode_id))
        frame_numbers = [ parse_frame_number(fname) for fname in frame_fnames ]
        terminal_frame = max(frame_numbers)
        for annotation in annotations["visual_results"]:
            start_frame, end_frame, labels, bboxes = parse_annotation(annotation)

            start_end_frames = []
            n_frames = end_frame + 1 - start_frame
            if n_frames >= C.n_frames_per_clip:
                for median_frame in range(start_frame + C.n_front, end_frame - C.n_back + 1, C.n_frames_per_clip):
                    start_frame, end_frame = get_endpoints_from_median_frame(median_frame)
                    if start_frame < 1: continue
                    if end_frame > terminal_frame: continue
                    start_end_frames.append(( start_frame, end_frame ))

            for start_frame, end_frame in start_end_frames:
                for label, bbox in zip(labels, bboxes):
                    if label == "": continue
                    total_list.append(( episode_id, start_frame, end_frame, [ label ], bbox ))
    total_list = merge_duplicates(total_list)
    return total_list


def split_list(total_list):
    # Get the indices of total list according to an action index.
    actionLabel_listIdxs_dict = { label: [] for label in C.action_labels }
    for i, d in enumerate(total_list):
        _, _, _, labels, _ = d
        for label in labels:
            actionLabel_listIdxs_dict[label].append(i)

    # Sort action labels along its n_clips.
    sorted_actions = sorted(C.action_labels, key=lambda l: C.rep2sta[C.idx2rep[str(l)]]["n_clips"])

    already_taken = [ False for _ in range(len(total_list)) ]
    train_idxs = []
    test_idxs = []
    for action in sorted_actions:
        list_idxs = actionLabel_listIdxs_dict[action]
        list_idxs = [ idx for idx in list_idxs if not already_taken[idx] ]
        random.shuffle(list_idxs)
        n_train = int(len(list_idxs) * C.train_ratio)
        train = list_idxs[:n_train]
        test = list_idxs[n_train:]

        for i in train:
            train_idxs.append(i)
            already_taken[i] = True
        for i in test:
            test_idxs.append(i)
            already_taken[i] = True
    assert all(already_taken)

    total_list = [ [ ep, str(s), str(e), ','.join(ls), ','.join(bb) ] for ep, s, e, ls, bb in total_list ]
    train_list = [ '\t'.join(total_list[i]) for i in train_idxs ]
    test_list = [ '\t'.join(total_list[i]) for i in test_idxs ]
    return train_list, test_list


def generate_train_test_list():
    os.makedirs(C.list_dpath, exist_ok=True)

    print("Generating a training and a test list...")
    total_list = parse_total_list()
    train_list, test_list = split_list(total_list)

    with open(C.train_list_fpath, 'w') as fout:
        fout.write('\n'.join(train_list))
    with open(C.test_list_fpath, 'w') as fout:
        fout.write('\n'.join(test_list))


if __name__ == "__main__":
    generate_train_test_list()

