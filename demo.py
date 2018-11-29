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
    height_block = h // 8
    margin_left = 40
    text_width = 350
    bar_width = 450
    frame = cv2.copyMakeBorder(frame, 0, 0, 0, pane_width, cv2.BORDER_CONSTANT, None, (0, 0, 0))
    frame = cv2.putText(
        frame,
        text="Ground truth: {}".format("-" if len(ground_truths) == 0 else ", ".join(ground_truths)),
        org=(w + margin_left, height_block),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=1,
        color=(0, 255, 0),
        thickness=3)
    sorted_actions = sorted(actions, key=lambda e: -e[1])
    for i, (action, score) in enumerate(sorted_actions, 3):
        if score < 0.01: break
        high_probability = score > 0.5

        frame = cv2.putText(
            frame,
            text=action,
            org=(w + margin_left, height_block * i),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if high_probability else (255, 255, 255),
            thickness=3)
        frame = cv2.rectangle(
            frame,
            pt1=( w + margin_left + text_width, int(height_block * (i - 0.35)) ),
            pt2=( w + margin_left + text_width + bar_width, int(height_block * (i + 0.05)) ),
            color=(0, 0, 255) if high_probability else (255, 255, 255))
        frame = cv2.rectangle(
            frame,
            pt1=( w + margin_left + text_width, int(height_block * (i - 0.35)) ),
            pt2=( w + margin_left + text_width + int(bar_width * score), int(height_block * (i + 0.05)) ),
            color=(0, 0, 255) if high_probability else (255, 255, 255),
            thickness=cv2.FILLED)
        frame = cv2.putText(
            frame,
            text="{:.2f}".format(score),
            org=(w + margin_left + text_width + bar_width + margin_left, height_block * i),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(0, 0, 255) if high_probability else (255, 255, 255),
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
        frame = generate_frame(frame, ground_truths, actions, pane_width)
        vout.write(frame)
    vout.release()


def generate_demos():
    pbar = tqdm(total=sum([ len(episodes) for episodes in C.episodes_list ]))
    for season, episodes in zip(C.seasons, C.episodes_list):
        for episode in episodes:
            pbar.set_description("Generating a demo video for S{:02}_EP{:02d}".format(season, episode))

            os.makedirs(C.demo_dpath, exist_ok=True)
            generate_demo(season, episode)

            pbar.update(1)


if __name__ == '__main__':
    generate_demos()

