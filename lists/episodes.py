"""
Format:
    S01_EP01    13  17    4, 7, 8, 9    0, 0, 0, 0
    S01_EP01    13  17    4, 7, 8, 9    123, 234, 345, 456
    ...
"""

import os

from tqdm import tqdm

from lists.utils import load_annotation, parse_frame_number, timestr_to_seconds, get_endpoints_from_median_frame, \
                        merge_duplicates
from config import ListConfig as C


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
        else:
            label = ""
            bbox = []
        labels.append(label)
        bboxes.append(bbox)

    return start_frame, end_frame, labels, bboxes


def fill_empty_frames(data):
    sorted_data = sorted(data, key=lambda d: int(d[1]))

    prev_frame_number = None
    filled_data = []
    for d in sorted_data:
        episode_id, frame_number, _, _, _ = d
        frame_number = int(frame_number)

        if prev_frame_number is not None and prev_frame_number + 1 != frame_number:
            for inter_frame_number in range(prev_frame_number + 1, frame_number):
                filled_data.append((
                    episode_id,
                    inter_frame_number,
                    inter_frame_number + C.n_frames_per_clip - 1,
                    '',
                    [ str(c) for c in [ 0, 0, C.full_shape['width'], C.full_shape['height'] ] ]  ))
        filled_data.append(d)
        prev_frame_number = frame_number
    return filled_data


def get_episode_list(season, episode):
    """ List up every frame """

    annotations = load_annotation(season, episode)

    frame_fnames = os.listdir(os.path.join("data/friends_trimmed/frames", "S{:02d}_EP{:02d}".format(season, episode)))
    frame_numbers = [ parse_frame_number(fname) for fname in frame_fnames ]
    terminal_frame = max(frame_numbers)

    episode_list = []
    for i in range(1, len(annotations['visual_results']), 3):
        annotation = annotations['visual_results'][i]
        start_frame, end_frame, labels, bboxes = parse_annotation(annotation)
        for median_frame in range(start_frame, end_frame):
            start, end = get_endpoints_from_median_frame(median_frame)
            if start < 1: continue
            if end > terminal_frame: continue

            episode_id = "S{:02d}_EP{:02d}".format(season, episode)
            for label, bbox in zip(labels, bboxes):
                episode_list.append(( episode_id, start, end, [ label ], bbox ))
    episode_list = merge_duplicates(episode_list)
    episode_list = fill_empty_frames(episode_list)
    episode_list = [ [ ep, str(s), str(e), ','.join(ls), ','.join(bb) ] for ep, s, e, ls, bb in episode_list ]
    episode_list = sorted(episode_list, key=lambda l: int(l[1]))
    episode_list = [ '\t'.join(l) for l in episode_list ]
    return episode_list


def generate_episodes_list():
    os.makedirs(C.list_dpath, exist_ok=True)

    pbar = tqdm(total=sum([ len(episodes) for episodes in C.episodes_list ]))
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            pbar.set_description("Generating a list from S{:02d}_EP{:02d}...".format(season, episode))

            episode_list = get_episode_list(season, episode)

            episode_list_fpath = C.list_fpath_tpl.format(season, episode)
            with open(episode_list_fpath, 'w') as fout:
                fout.write('\n'.join(episode_list))

            pbar.update(1)


if __name__ == "__main__":
    generate_episodes_list()

