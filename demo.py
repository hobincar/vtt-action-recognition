from collections import defaultdict
import json
import os

import cv2
from tqdm import tqdm

from config import DemoConfig as C


def load_frame(season, episode, frame_number):
    frame_fpath = C.frame_fpath_tpl.format(season, episode, frame_number)
    frame = cv2.imread(frame_fpath)
    return frame


def generate_frame(frame, ground_truths, actions, pane_width):
    h, w, c = frame.shape

    pane_width = 1000
    block_height = h // 8
    margin_left = 40
    text_width = 350
    bar_width = 450
    frame = cv2.copyMakeBorder(frame, 0, 0, 0, pane_width, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    frame = cv2.putText(
        frame,
        text="Ground truth: {}".format("-" if len(ground_truths) == 0 else ", ".join(ground_truths)),
        org=(w + margin_left, block_height),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=3)
    sorted_actions = sorted(actions, key=lambda e: -e[1])
    for i, (action, score) in enumerate(sorted_actions, 3):
        if score < 0.01: break
        high_probability = score > C.high_prob_threshold

        frame = cv2.putText(
            frame,
            text=action,
            org=(w + margin_left, block_height * i),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if high_probability else (255, 255, 255),
            thickness=3)
        frame = cv2.rectangle(
            frame,
            pt1=( w + margin_left + text_width, int(block_height * (i - 0.35)) ),
            pt2=( w + margin_left + text_width + bar_width, int(block_height * (i + 0.05)) ),
            color=(0, 0, 255) if high_probability else (255, 255, 255))
        frame = cv2.rectangle(
            frame,
            pt1=( w + margin_left + text_width, int(block_height * (i - 0.35)) ),
            pt2=( w + margin_left + text_width + int(bar_width * score), int(block_height * (i + 0.05)) ),
            color=(0, 0, 255) if high_probability else (255, 255, 255),
            thickness=cv2.FILLED)
        frame = cv2.putText(
            frame,
            text="{:.2f}".format(score),
            org=(w + margin_left + text_width + bar_width + margin_left, block_height * i),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if high_probability else (255, 255, 255),
            thickness=3)
    return frame


def generate_frame_with_bbox(frame, ground_truths_list, actions_list, bbox_list, pane_width):
    h, w, c = frame.shape

    pane_width = 1000
    block_height = h // 9
    margin_left = 40
    text_width = 350
    bar_width = 450
    frame = cv2.copyMakeBorder(frame, 0, 0, 0, pane_width, cv2.BORDER_CONSTANT, None, (0, 0, 0))

    n_data = len(ground_truths_list)
    n_block_per_char = [ None, 5, 3, 2, 1, 1, 1 ][n_data]
    colors = [ (0, 0, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 255, 255), (128, 255, 128) ]
    for k, (ground_truths, actions, bbox, theme_color) in enumerate(zip(ground_truths_list, actions_list, bbox_list, colors)):
        top = k * n_block_per_char * block_height
        left = w + margin_left
        right = w + pane_width
        bottom = (k + 1) * n_block_per_char * block_height

        x1, y1, x2, y2 = bbox
        frame = cv2.rectangle(
            frame,
            pt1=( x1, y1 ),
            pt2=( x2, y2 ),
            color=theme_color,
            thickness=5)
        frame = cv2.rectangle(
            frame,
            pt1=( left - 5, top + 35 ),
            pt2=( right - 5, bottom + 30 ),
            color=theme_color,
            thickness=5)
        frame = cv2.putText(
            frame,
            text="Ground truth: {}".format("-" if len(ground_truths) == 0 else ", ".join(ground_truths)),
            org=(left, top + 1 * block_height),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 255, 0),
            thickness=3)

        sorted_actions = sorted(actions, key=lambda e: -e[1])
        sorted_actions = sorted_actions[:n_block_per_char - 1]
        for i, (action, score) in enumerate(sorted_actions, 2):
            if score < 0.01: break
            high_probability = score > C.high_prob_threshold

            frame = cv2.putText(
                frame,
                text=action,
                org=(left, top + i * block_height),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=(0, 0, 255) if high_probability else (255, 255, 255),
                thickness=3)
            frame = cv2.rectangle(
                frame,
                pt1=( left + text_width, top + int((i - 0.35) * block_height) ),
                pt2=( left + text_width + bar_width, top + int((i + 0.05) * block_height) ),
                color=theme_color if high_probability else (255, 255, 255))
            frame = cv2.rectangle(
                frame,
                pt1=( left + text_width, top + int((i - 0.35) * block_height) ),
                pt2=( left + text_width + int(bar_width * score), top + int((i + 0.05) * block_height) ),
                color=theme_color if high_probability else (255, 255, 255),
                thickness=cv2.FILLED)
            frame = cv2.putText(
                frame,
                text="{:.2f}".format(score),
                org=(left + text_width + bar_width + margin_left, top + i * block_height),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=theme_color if high_probability else (255, 255, 255),
                thickness=3)
    return frame


def generate_demo(season, episode):
    episode_id = "S{:02d}_EP{:02d}".format(season, episode)
    prediction_fpath = C.prediction_fpath_tpl.format(season, episode)
    with open(prediction_fpath, 'r') as fin:
        prediction = json.load(fin)
        prediction_results = prediction["prediction_results"]

    tmp_frame_number =  prediction_results[0]["frame"]
    tmp_frame_fpath = C.frame_fpath_tpl.format(season, episode, tmp_frame_number)
    tmp_frame = cv2.imread(tmp_frame_fpath)
    height, width, layers = tmp_frame.shape

    demo_fpath = C.demo_fpath_tpl.format(season, episode)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    pane_width = 1000
    vout = cv2.VideoWriter(demo_fpath, apiPreference=0, fourcc=fourcc, fps=5, frameSize=(width + pane_width, height))
    for pred in prediction_results:
        frame_number = pred["frame"]
        ground_truths = pred["ground_truths"]
        actions = pred["actions"]

        frame = load_frame(season, episode, frame_number)

        """ TEMP """
        bbox_fpath = "data/friends_json/bbox/person/S{:02d}_EP{:02d}/{:05d}.json".format(season, episode, frame_number)
        with open(bbox_fpath, 'r') as fin:
            bboxes = json.load(fin)
        bboxes = [ bbox for bbox in bboxes if bbox['confidence'] > 0.5 and bbox['label'] == 'person' ]
        for bbox in bboxes:
            x1, y1 = bbox['topleft']['x'], bbox['topleft']['y']
            x2, y2 = bbox['bottomright']['x'], bbox['bottomright']['y']
            frame = cv2.rectangle(
                frame,
                pt1=( x1, y1 ),
                pt2=( x2, y2 ),
                color=(0, 255, 255),
                thickness=5)
        """ TEMP """
        frame = generate_frame(frame, ground_truths, actions, pane_width)
        vout.write(frame)
    vout.release()


def __merge(lst):
    frame_number = lst[0]['frame']

    merged = {
        'frame': frame_number,
        'ground_truths_list': [],
        'actions_list': [],
        'bbox_list': [],
    }
    for l in lst:
        merged['ground_truths_list'].append(l['ground_truths'])
        merged['actions_list'].append(l['actions'])
        merged['bbox_list'].append(l['bbox'])
    return merged


def merge_along_frame(data):
    collected_data = defaultdict(lambda: [])
    for d in data:
        frame_number = d['frame']
        collected_data[frame_number].append(d)

    merged_data = list(map(__merge, collected_data.values()))

    return merged_data


def generate_demo_with_bbox(season, episode):
    episode_id = "S{:02d}_EP{:02d}".format(season, episode)
    prediction_fpath = C.prediction_fpath_tpl.format(season, episode)
    with open(prediction_fpath, 'r') as fin:
        prediction = json.load(fin)
        prediction_results = prediction["prediction_results"]

    tmp_frame_number =  prediction_results[0]["frame"]
    tmp_frame_fpath = C.frame_fpath_tpl.format(season, episode, tmp_frame_number)
    tmp_frame = cv2.imread(tmp_frame_fpath)
    height, width, layers = tmp_frame.shape

    demo_fpath = C.demo_fpath_tpl.format(season, episode)
    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    pane_width = 1000
    vout = cv2.VideoWriter(demo_fpath, apiPreference=0, fourcc=fourcc, fps=5, frameSize=(width + pane_width, height))

    prediction_results = merge_along_frame(prediction_results)

    for pred in prediction_results:
        frame_number = pred["frame"]
        ground_truths_list = pred["ground_truths_list"]
        actions_list = pred["actions_list"]
        bbox_list = pred["bbox_list"]

        frame = load_frame(season, episode, frame_number)
        frame = generate_frame_with_bbox(frame, ground_truths_list, actions_list, bbox_list, pane_width)
        vout.write(frame)
    vout.release()


def generate_demos():
    pbar = tqdm(total=sum([ len(episodes) for episodes in C.episodes_list ]))
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            pbar.set_description("Generating a demo video for S{:02}_EP{:02d}".format(season, episode))

            os.makedirs(os.path.dirname(C.demo_fpath_tpl), exist_ok=True)
            if C.use_bbox:
                generate_demo_with_bbox(season, episode)
            else:
                generate_demo(season, episode)

            pbar.update(1)


if __name__ == '__main__':
    generate_demos()

