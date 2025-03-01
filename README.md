# QuickAnnotator
---
Quick Annotator is an open-source digital pathology annotation tool.

![QA user interface screenshot](https://github.com/choosehappy/QuickAnnotator/wiki/images/Annotation_Page_LayerSwitch.gif)

# Purpose
---
Machine learning approaches for segmentation of histologic primitives (e.g., cell nuclei) in digital 
pathology (DP) Whole Slide Images (WSI) require large numbers of exemplars. Unfortunately, annotating 
each object is laborious and often intractable even in moderately sized cohorts. 
The purpose of the quick annotator is to rapidly bootstrap annotation creation for digital
pathology projects by helping identify images and small regions.

Because the classifier is likely to struggle by intentionally focusing in these areas, 
less pixel level annotations are needed, which significantly improves the efficiency of the users.
Our approach involves updating a u-net model while the user provides annotations, so the model is
then in real time used to produce. This allows the user to either accept or modify regions of the 
prediction.

DADA AL minimizes the number of annotated patches needed to train a deep learning model. In this fork of QA, 
DADA is integrated in the user interface so that an annotator can quickly build a training set for different
applications. So far, TIL presence in patches is the focus of the present work, which can also be applied to
other scenarios.

# Requirements
---
Tested with Python 3.7 and 3.8

Requires:
1. Python 
2. pip
3. DADA source
4. Tensorflow-gpu (for AL)
5. Openslide C library

And the following additional python package:
1. Flask_SQLAlchemy
2. scikit_image
3. scikit_learn
4. opencv_python_headless
5. scipy
6. requests
7. SQLAlchemy
8. torch
9. torchvision
10.Flask_Restless
11. numpy
12. Flask
13. umap_learn
14. Pillow
15. tensorboard
16. ttach
17. albumentations
18. config
19. Pandas
20. TQDM
21. Imageio
22. Imagesize
23. Imgaug
24. Keras with Keras contrib

# Installation
It is highly recommended that instalation takes place inside a Python3 Virtual environment, so create one with:
 ```
 python3 -m venv VENV_DIR
 ```

1. Clone current repository
2. Clone DADA AL from: https://github.com/alsmeirelles/DADA
3. Start virtual environment with:
 ```
 source VENV_DIR/bin/activate
  ```
 4. Install Tensorflow
 - If a GPU is available:
 ```
pip3 install tensorflow-gpu==1.15.5
```
- If not:
 ```
 pip3 install tensorflow==1.15.5
 ```
 5. Install Openslide C library (available in multiple distribution package formats https://openslide.org/download/)
 6. Install other requirements through requirements.txt file:

 ```
 pip3 install -r requirements.txt
 ```
 7. Edit configuration file, as described bellow


*Note:* DADA AL was tested with *cuda version 10*.

The library versions have been pegged to the current validated ones. 
Later versions are likely to work but may not allow for cross-site/version reproducibility

We received some feedback that users could installed *torch*. Here, we provide a detailed guide to install
*Torch*

### Torch's Installation
The general guides for installing Pytorch can be summarized as following:
1. Check your NVIDIA GPU Compute Capability @ *https://developer.nvidia.com/cuda-gpus* 
2. Download CUDA Toolkit @ *https://developer.nvidia.com/cuda-downloads* 
3. Install PyTorch command can be found @ *https://pytorch.org/get-started/locally/* 

# Basic Usage
---
see [UserManual](https://github.com/choosehappy/QuickAnnotator/wiki/User-Manual) for a demo

### Creating a patch pool
WSIs involved in the experiment must first be split into patches.

A script to execute patch extraction from slides is available in the Utils folder of DADA source tree and can be run as so:
```
python3 WSITile.py -ds path_to_slides_folder -od patch_destination_folder 
```
Optionally, patch size can be defined with the -ps option. A multiproccess extraction can be done with the -mp option (i.e -mp 4).

### Run

Go to your checkout dir and start the web server:
```
python QA.py
```
By default, it will start up on *127.0.0.1:5555*. Note that *5555* is the port number setting in [config.ini](https://github.com/choosehappy/QuickAnnotator/blob/main/config/config.ini#L6) and user should confirm {port number} is not pre-occupied by other users on the host. 

*Warning*: virtualenv will not work with paths that have spaces in them, so make sure the entire path to `env/` is free of spaces.

### Config Sections
There are many modular functions in QA whose behaviors could be adjusted by hyper-parameters. These hyper-parameters can 
be set in the *config.ini* file, inside config dir
- [common]
- [active_learning]
- [flask]
- [cuda]
- [sqlalchemy]
- [pooling]
- [train_ae]
- [train_tl]
- [make_patches]
- [make_embed]
- [get_prediction]
- [frontend]
- [superpixel]

Some configuration parameters should be defined by the user before he/she can start using the interface. These are:
1. In *common*:
- wsis = Path to where the slides are located
- pool = Path to the pool of patches extracted from the slides

2. In *active_learning*:
- alsource = Path to DADA source code
- strategy = One of EnsembleALTrainer (ensemble approach) or ActiveLearningTrainer (MC Dropout)
- un_function = Uncertainty function to use, which should correspond to selected strategy (see DADA repository docs)
- dropout_steps = number of forward passes for MC Dropout strategy
- alepochs = number of epochs to train each intermediary model (Ex: 50 epochs)
- phi = network auto-reduction (NAR) coefficient, higher means faster selection with possible quality reduction

# Advanced Usage
---
See [wiki](https://github.com/choosehappy/QuickAnnotator/wiki)

# Citation
---
Read the related paper on arXiv: [Quick Annotator: an open-source digital pathology based rapid image annotation tool](https://arxiv.org/abs/2101.02183)

PDF file available for [download](https://arxiv.org/ftp/arxiv/papers/2101/2101.02183.pdf)

Please use below to cite this paper if you find this repository useful or if you use the software shared here in your research.
```
  @misc{miao2021quick,
      title={Quick Annotator: an open-source digital pathology based rapid image annotation tool}, 
      author={Runtian Miao and Robert Toth and Yu Zhou and Anant Madabhushi and Andrew Janowczyk},
      year={2021},
      eprint={2101.02183},
      archivePrefix={arXiv},
      primaryClass={eess.IV}
  }
```
# Frequently Asked Questions
See [FAQ](https://github.com/choosehappy/QuickAnnotator/wiki/Frequently-Asked-Questions)




