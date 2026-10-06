# Machine Learning Techniques for Distracted Driver Detection

## Project Overview

Distracted driving is a major road-safety concern because activities such as texting, talking on the phone, drinking, operating the radio, or interacting with passengers can reduce a driver's attention.

This project focuses on detecting different driver behaviors from images using machine learning and deep learning techniques.

The project uses the **State Farm Distracted Driver Detection dataset**, which contains images of drivers performing different activities. The task is formulated as a **10-class image classification problem**.

Two approaches were implemented and compared:

1. **Person 1:** A custom Convolutional Neural Network (CNN) was developed as the baseline model.
2. **Person 2:** Transfer learning using an ImageNet-pretrained **ResNet-18** model was implemented as an improved approach.

The main objective of the project is to determine whether transfer learning can improve distracted-driver classification performance compared with a custom CNN trained for the same task.

---

# Problem Statement

The objective of this project is to automatically classify a driver's behavior from an input image.

Given an image of a driver, the model predicts one of ten predefined driver-behavior classes.

The ten classes represent safe driving and different forms of distracted driving, including texting, phone usage, drinking, operating the radio, reaching behind, hair and makeup, and talking to a passenger.

The problem is therefore treated as a **multi-class image classification problem with 10 classes**.

---

# Dataset

The project uses the **State Farm Distracted Driver Detection dataset**.

The dataset contains approximately **22,424 training images** distributed across ten driver-behavior classes.

The dataset is not included in this repository because of its large size.

## Dataset Classes

| Class | Behavior |
|---|---|
| c0 | Safe driving |
| c1 | Texting right |
| c2 | Phone right |
| c3 | Texting left |
| c4 | Phone left |
| c5 | Operating radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

## Dataset Distribution

| Class | Behavior | Number of Images |
|---|---|---:|
| c0 | Safe driving | 2489 |
| c1 | Texting right | 2267 |
| c2 | Phone right | 2317 |
| c3 | Texting left | 2346 |
| c4 | Phone left | 2326 |
| c5 | Operating radio | 2312 |
| c6 | Drinking | 2325 |
| c7 | Reaching behind | 2002 |
| c8 | Hair and makeup | 1911 |
| c9 | Talking to passenger | 2129 |
| **Total** | | **22,424** |

---

# Dataset Organization

The dataset is organized into ten class folders.

The expected directory structure is:

```text
imgs/
└── train/
    ├── c0/
    ├── c1/
    ├── c2/
    ├── c3/
    ├── c4/
    ├── c5/
    ├── c6/
    ├── c7/
    ├── c8/
    └── c9/
