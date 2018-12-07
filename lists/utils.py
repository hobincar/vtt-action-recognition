from collections import defaultdict
import json

import parse

from config import ListConfig as C


def load_annotation(season, episode):
    annotation_fpath = C.annotation_fpath_tpl.format(season, episode)
    with open(annotation_fpath, 'r') as fin:
        annotation = json.load(fin)
    return annotation


def parse_frame_number(fname):
    p = parse.compile("{:d}.jpg")
    frame_number = p.parse(fname)[0]
    return frame_number


def timestr_to_seconds(timestr):
    time_parser = parse.compile("{:d}:{:d}:{:d};{:d}")
    h, m, s, ms = time_parser.parse(timestr)
    seconds = 3600*h + 60*m + s + 1/60*ms
    return seconds


def get_endpoints_from_median_frame(median_frame):
    start_frame = median_frame - C.n_front
    end_frame = median_frame + C.n_back
    return start_frame, end_frame


def __merge(dups):
    episode_id, start_frame, end_frame, _, bbox = dups[0]
    labels = []
    for dup_episode_id, dup_start_frame, dup_end_frame, dup_labels, dup_bbox in dups:
        assert dup_episode_id == episode_id
        assert dup_start_frame == start_frame
        assert dup_end_frame == end_frame
        assert ','.join(dup_bbox) == ','.join(bbox)
        labels += dup_labels
    labels = list(set(labels))
    return ( episode_id, start_frame, end_frame, labels, bbox )


def merge_duplicates(data):
    merge_dict = defaultdict(lambda: [])
    for d in data:
        episode_id, start_frame, _, _, bbox = d
        if len(bbox) == 0: continue
        id = '_'.join([ episode_id, str(start_frame), ','.join(bbox) ])
        merge_dict[id].append(d)

    merged_list = list(map(__merge, merge_dict.values()))
    return merged_list

