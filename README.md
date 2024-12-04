# Term7_ComputerVision_Project

Theme: Design Track - healthcare

Initial project idea: Utilizing human pose estimation models to determine whether someone is exercising correctly

Background: Human pose estimation, a subfield of computer vision, involves the task of localizing and identifying human body joints in an image or video sequence. This technology has seen significant advancements in recent years, driven by improvements in deep learning algorithms and the availability of large-scale datasets. Our group wishes to use this existing technology to help people exercise better by tracking landmarks on their body to determine whether their posture is correct when doing a particular exercise. 

This task is a complex computer vision task because the problem deals with object occlusion, lighting conditions, body position and different clothing. Our group will be using GoogleMediaPipe which uses the BlazePose model for 3D pose estimation. By identifying key landmarks on the body, such as the shoulders, elbows, wrists, hips, knees, and ankles, we can assess the correctness of an individual's posture while performing various exercises.

Another challenge would be processing images in real time as we have to process frames very quickly determining the landmark positions per frame and providing real time feedback. 

The following link is the blazepose research paper:  https://arxiv.org/abs/2006.10204. 

## Implementation

This is a third-party implementation of BlazePose in Tensorflow from [alishsuper/BlazePose-Implementation](https://github.com/alishsuper/BlazePose-Implementation?tab=readme-ov-file). 

### Setup

1. Install required libraries:
    ```
    python -m venv .venv
    .venv\Scripts\activate
    python -m pip install -r requirements.txt
    ```

2. Run `python test.py`. The keypoint predictions will be shown in the `/results` folder.

### Train/finetune model

In `config.py`, modify `use_existing_model_weights = 0`. Then refer to `README_OG.md`.