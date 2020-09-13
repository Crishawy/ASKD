# ASKD
This repo is the implementation of paper `Attribute Structure Knowledge Distillation`.


## Requirement
This repo is tested with Ubuntu 18.04.4, Python 3.8.3, PyTorch 1.6.0, CUDA 10.2.
Make sure to install pytorch, torchvision, yacs, numpy before using this repo.

## Running
The experiments are run by configuring yaml files. 
### Teacher Training
An example of training `resnet32x4` vanilla teacher is:
```
python train_teacher.py -cfg ./configs/train_vanilla/resnet32x4.yaml
```
where you can configure the architecture via modifying the `model` property in yaml file.

You can also download all the pre-trained teacher models on [baiduyun]()
or [google cloud](). 

### Student Training
An example of training `shufflev2` taught by `resnet32x4` with ASKD is:
```
python train_student.py -cfg ./configs/kds/ours/res32x4_shufflenetv2.yaml
```
The training hyper-parameters are configured in yaml file. All the yaml files of 
different compared distillation methods are available in `./configs/kds`.

## Results (Top-1 Acc) on CIFAR100

### Similar Architecture

| Teacher <br> Student | resnet110 <br> resnet32 | resnet110 <br> resnet20 | resnet32x4 <br> resnet8x4 | wrn-40-2 <br> wrn-16-2 |  wrn-40-2 <br> wrn-40-1 |
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:-----------:|
| Teacher <br> Student |    74.31 <br> 71.25    |    74.31 <br> 69.31    |    79.42 <br> 72.83    |     75.61 <br> 73.46     | 75.61 <br> 72.38 |
| HKD | 73.73 | 71.32 | 74.54 | 74.99 | 74.08 |
| FitNet | 71.04 | 69.51 | 73.47 | 73.63 | 72.46 |
| AT | 72.71 | 70.55 | 73.45 | 74.00 | 72.46 |
| FSP | 71.76 | 70.11 | 72.57 | 73.35 | N/A |
| RKD | 72.60 | 69.57 | 72.05 | 73.32 | 71.62 |
| CC | 72.40 | 69.89 | 73.09 | 73.57 | 71.70 |
| SP | 72.53 | 70.57 | 73.04 | 73.88 | 72.83 |
| CRD | 73.59 | 71.34 | 75.21 | **75.91** | 73.98 |
| **ASKD** | **73.85** | **71.54** | **75.34** | 75.71 | **74.33** |

### Different Architecture

| Teacher <br> Student | vgg13 <br> MobieleNetV2 | ResNet50 <br> MobileNetV2 | ResNet50 <br> vgg8 | resnet32x4 <br> ShuffleV1 |  resnet32x4 <br> ShuffleV2 | wrn40-2 <br> ShuffleV1|
|:---------------:|:-----------------:|:-----------------:|:-----------------:|:--------------------:|:-----------:|:-------------:|
| Teacher <br> Student |    74.64 <br> 65.45    |    79.34 <br> 65.45    |    79.34 <br> 70.25    |    79.42 <br> 71.57     | 79.42 <br> 73.35 | 75.61 <br> 71.57 |
| HKD | 68.01 | 68.74| 73.56| 74.51| 75.52| 75.53|
| FitNet |65.09 | 62.79 | 69.71 | 74.05 | 75.00 | 74.29 |
| AT | 59.50 | 57.61 | 72.04 | 73.14 | 73.48 | 74.65 |
| RKD | 65.10 | 73.42 | 71.10 | 73.09 | 74.10 | 73.11 |
| CC | 64.91 | 65.58 | 70.77 | 71.55 | 73.12 | 71.60 |
| SP | 67.02 | 67.86 | 73.25 | 76.04 | 76.10 | 75.90 |
| CRD | **69.02** | 69.18 | 74.41 | 75.75 | 75.95 | 75.82 |
| **ASKD** | 68.94 | **69.73** | **74.95** | **77.27** | **77.06** | **76.70** |

## Acknowledgement
The repo is based on [CRD](https://github.com/HobbitLong/RepDistiller).
