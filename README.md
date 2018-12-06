# VTT-action-recognition

This project tries to recognize actions occuring in the American television sitcom, *Friends*. Original code is from [C3D-tensorflow](https://github.com/hx173149/C3D-tensorflow), and I modified it for this project.

And I borrowed some codes which implement following networks

* [C3D](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf) from [C3D-tensorflow](https://github.com/hx173149/C3D-tensorflow) by [hx173149](https://github.com/hx173149)



# Requirements

* Ubuntu 16.04
* CUDA 9.0
* cuDNN 7.3.1
* Python 3.6.6
* Python libraries listed on requirements.txt (including tensorflow 1.12.0)



# How to use


## Step 1. Prepare Data

For this project, you need to have following two data.
* Frames of each *Friends* video
  1. Download *Friends* videos.
  2. Extract frames from each video on 5 fps, and locate them on 
     `<project_root>/data/friends_trimmed/frames/S<season>_EP<episode>/<frame_number>.jpg`
  
     e.g. `<project_root>/data/friends_trimmed/frames/S01_EP01/00001.jpg`

* Annotations
  1. Download two annotation files we got from Konan Technology
     * `VTT3_2차년도_메타데이터1차_배포_20180809.zip`
     * `20181024_VTT3세부_메타데이터_2차배포.zip`
  2. Extract following json files 
     * `s<season>_ep<episode>_tag2_visual_Final_180809.json` from `VTT3_2차년도_메타데이터1차_배포_20180809.zip`
     * `s<season>_ep<episode>_tag2_visual_final.json` from `20181024_VTT3세부_메타데이터_2차배포.zip`
     
     and locate the json files on the following directory
     
     `<project_root>/data/friends_trimmed/annotations`

  So, the directory structure will be
  ```
  data/
    friends_trimmed/
      frames/
        S01_EP01/
          00001.jpg
          ...
        ...
      annotations/
        S01_EP01.json
        ...
  ```


## Step 2. Analyze Data

By combining some duplicated action classes like *Standing up*, *standing up*, *stading up* into *standing*, I can have 32 action classes. After removing *none* which indicates *no action* and trimming action classes which is assigned to only 1 clip (*cup*, *' '* (blank), *desk*), I can have 28 action classes as following.

![image](https://user-images.githubusercontent.com/17702664/49156984-0e79ad80-f362-11e8-8e0f-a17c4ba2f083.png)


## Step 3. Construct dataset from the annotation files

Each ground truth annotation have information of "start seconds", "end seconds", and "action classes".

```
{
  "start seconds": "00:01:46;16",
  "end seconds": "00:01:51;60",
  "actions": [ "Holding something", "standing" ]
}
```

So, I constructed dataset through following steps.

1. Since I extracted 5 frames per second from each video, I can get the frame number of each seconds.
   
   e.g.
   ```
   {
     "start seconds": frame #531,
     "end seconds": frame #558,
     "actions": [ "Holding something", "standing" ]
   }
   ```

2. As specified in [C3D paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.pdf), I extracted every 16 frames which are overlapped 8 frames.
   
   e.g.
   ```
   [
     {
       "start seconds": frame #531,
       "end seconds": frame #546,
       "actions": [ "Holding something", "standing" ]
     },
     {
       "start seconds": frame #546,
       "end seconds": frame #561,
       "actions": [ "Holding something", "standing" ]
     },
     {
       "start seconds": frame #561,
       "end seconds": frame #576,
       "actions": [ "Holding something", "standing" ]
     }
   ]
   ```
      
   If the length of clip is less than 16 frames, calculate the median between the start and the end and pad with some frames around them (so it'll contain some frames that does not belong to the groundtruth actions).
   
   ```
   {
     "start seconds": the median - 7,
     "end seconds": the median + 8,
     "actions": [ "Holding something", "standing" ]
   }
   ```


## Step 4. Split dataset into train & test

```
$ python -m lists.train_test
```

I divided the dataset into training and testing with balanced label distribution in mind. Since this project deals with a multi-label classification task, I cannot strictly divide all labels into a train and a test dataset with same ratio. So I tried to more fairly divide data which has more few clips.

![image](https://user-images.githubusercontent.com/17702664/49162055-72ee3a00-f36d-11e8-99d3-81ea6431369d.png)


## Step 5. Train

```
$ CUDA_VISIBLE_DEVICES=0 python train.py
```

*NOTE: Multi-gpu training is not supported yet.*

* Result
  
  | Base model | Precision | Recall | F1 score |
  | :---: | :----: | :---: | :---: |
  | C3D | 0.5455 | 0.5162 | 0.5299 |
  
  ![image](https://user-images.githubusercontent.com/17702664/49209459-e2ac0580-f3fd-11e8-9b5a-39d6a85aac88.png)

* Examples
  
  ![image](https://user-images.githubusercontent.com/17702664/49210335-33bcf900-f400-11e8-9e91-399463e5eabd.png)
  
  ![image](https://user-images.githubusercontent.com/17702664/49210446-71ba1d00-f400-11e8-9981-4a5450582fc1.png)

## Step 6. Predict

  ```
  $ python -m lists.episodes
  $ CUDA_VISIBLE_DEVICES=0 python predict.py
  ```
  
  It will generate **json files** in `<project root>/outputs/predictions` used to generate demo videos, and **jsonlines files** in `<project root>/outputs/integration` for integration. The json schema follows one defined at https://github.com/uilab-vtt/knowledge-graph-input


## Step 7. Demo
  
  ```
  $ python demo.py
  ```
  
  By using files generated in **Step 6**, it will generate **demo videos** in `<project root>/outputs/demo" for each *friends* episode.


# References

* Tran, Du, et al. "Learning spatiotemporal features with 3d convolutional networks." Proceedings of the IEEE international conference on computer vision. 2015.


# Acknowledgements

This work was supported by Institute for Information & communications Technology Promotion(IITP) grant funded by the Korea government(MSIT) (2017-0-01780, The technology development for event recognition/relational reasoning and learning knowledge based system for video understanding)
