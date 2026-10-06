# Machine Learning Techniques for Distracted Driver Detection

## Project Overview

This project focuses on detecting distracted driving behaviors from driver images using machine learning and deep learning techniques.

The project uses the State Farm Distracted Driver Detection dataset containing 10 different driver behavior classes.

## Dataset

The dataset contains 10 classes:

| Class | Behavior |
|------|----------|
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

The dataset is not included in this repository because of its large size.

## Methods

### Person 1 — Baseline CNN

A custom Convolutional Neural Network (CNN) was developed as the baseline model.

The baseline achieved approximately:

- Validation Accuracy: **98.75%**
- Macro F1-score: **98.68%**

The main weakness observed was the **Hair and Makeup** class, which had lower recall compared with the other classes.

### Person 2 — Transfer Learning

A transfer learning approach was implemented using an **ImageNet-pretrained ResNet-18** model.

The same 10-class dataset and validation setup were used to make the comparison with the baseline fair.

The ResNet-18 model achieved:

- Validation Accuracy: **99.75%**
- Improvement over baseline: **+1.00 percentage point**

ResNet-18 also improved the overall macro precision, recall, and F1-score compared with the baseline CNN.

## Project Structure

```text
Distracted_driver/
│
├── models/
│   ├── distracted_driver_cnn.pth
│   ├── person2_resnet18.pth
│   └── person2_resnet18_best.pth
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_person2_transfer_learning.ipynb
│
├── results/
│   ├── classification_report.txt
│   ├── confusion_matrix/
│   └── plots/
│       ├── training_validation_accuracy.png
│       ├── training_validation_loss.png
│       ├── person2_training_validation_accuracy.png
│       └── person2_training_validation_loss.png
│
├── .gitignore
└── requirements.txt