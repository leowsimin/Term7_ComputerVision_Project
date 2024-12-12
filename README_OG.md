# BlazePose-Implementation in Tensorflow
BlazePose paper is "BlazePose: On-device Real-time Body Pose tracking" by Valentin Bazarevsky, Ivan Grishchenko, Karthik Raveendran, Tyler Zhu, Fan Zhang, and Matthias Grundmann. Available on [arXiv](https://arxiv.org/abs/2006.10204).

## Requirements
```
Anaconda
Python
Tensorflow
```
Please import Anaconda environment file (BlazePoseTensorflow.yml).

## Dataset
Leeds Sports Pose Dataset
Sam Johnson and Mark Everingham
http://sam.johnson.io/research/lsp.html

This dataset contains 2000 images of mostly sports people
gathered from Flickr. The images have been scaled such that the
most prominent person is roughly 150 pixels in length. The file
joints.mat contains 14 joint locations for each image along with
a binary value specifying joint visibility.

The ordering of the joints is as follows:
```
Right ankle
Right knee
Right hip
Left hip
Left knee
Left ankle
Right wrist
Right elbow
Right shoulder
Left shoulder
Left elbow
Left wrist
Neck
Head top
```

## Model
![image](https://user-images.githubusercontent.com/14852495/156509720-2d900f7b-8953-4219-9aa8-dea97dccb93c.png)
![image](https://user-images.githubusercontent.com/14852495/156510922-5d962d87-e021-4a3f-9c67-3afbd168a022.png)

## Train
1. Pre-train the heatmap branch and the regression branch.
  run `python train.py`.

## Test
1. Set `epoch_to_test` to the epoch you would like to test in config.py.

2. Run `python test.py`.

## Performance Comparison

Base Model PCK: 0.58
ViT Model PCK: 


## Reference

```tex
@article{Bazarevsky2020BlazePoseOR,
  title={BlazePose: On-device Real-time Body Pose tracking},
  author={Valentin Bazarevsky and I. Grishchenko and K. Raveendran and Tyler Lixuan Zhu and Fangfang Zhang and M. Grundmann},
  journal={ArXiv},
  year={2020},
  volume={abs/2006.10204}
}
```
