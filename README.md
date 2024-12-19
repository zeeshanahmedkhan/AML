# AMD-FV: Adaptive Margin Loss and Dual Path Network+ for Deep Face Verification

## Overview
The AMD-FV project implements the Adaptive Margin Loss (AML) and Dual Path Network+ (DPN+) architectures for face verification tasks. This repository contains the code, datasets (download paths), and scripts necessary to reproduce the experiments and results reported in the paper "AMD-FV: Adaptive Margin Loss and Dual Path Network+ for Deep Face Verification."

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Datasets](#datasets)
- [Results](#results)


## Features
- Implementation of the Adaptive Margin Loss (AML) for improved face verification accuracy.
- Dual Path Network+ (DPN+) architecture designed for efficient feature extraction.
- Comprehensive evaluation on standard face verification datasets.
- Scripts for training and testing the model.

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/zeeshanahmedkhan/AML.git
   cd AML

## Usage

2. To train the model, run AMD-FV/main/AMD.py

3. To validate the model, run AMD-FV/main/validate.py


## Datasets

The project utilizes several face verification datasets. Ensure you have the datasets downloaded and properly formatted before running the training or evaluation scripts. The datasets used in this project include:

   Training Dataset: MS1Mv2

   
   Test Datasets: LFW, Megaface, IJB-B, CALFW, and CPLFW

For more details on dataset preparation, please refer to the documentation in the respective scripts.

## Results

The experimental results demonstrate the effectiveness of the proposed AML and DPN+ methods, achieving state-of-the-art performance on the evaluated datasets. For detailed results, including accuracy metrics, refer to the results section of the paper.


