# Self-training Guided Adversarial Domain Adaptation For Thermal Imagery

<p align="center">
  <img src="images/activation_maps.png" width="400">
</p>

If you make use of this code, please cite the following paper:
```
Citation goes here.
```

## Overview
This repository contains official implementation of "[Self-training Guided Adversarial Domain Adaptation For Thermal Imagery](https://arxiv.org/abs/1801.07939)" paper (accepted to CVPR 2021 [Perception Beyond Visible Spectrum (PBVS)](https://pbvs-workshop.github.io/) workshop).

![](/images/sgada.png)

## Note
Before running the training code, make sure that `DATASETDIR` environment variable is set to your dataset directory.

## Environment
- Python 3.8.5
- PyTorch 1.6.0

To install the environment using Conda:
```
$ conda env create -f requirements_conda.yml
```

This command creates a Conda environment named `sgada`. The environment includes all necessary packages for training of SGADA method.

## Acknowledgement
This repo is mostly based on https://github.com/Fujiki-Nakamura/ADDA.PyTorch.
