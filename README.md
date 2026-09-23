# Oxford-IIIT Pet Breed Classification

## Project Overview

This project focuses on classifying images of cats and dogs into 37 different pet breeds using deep learning and computer vision.

The project compares a custom Convolutional Neural Network (CNN) with a pretrained ResNet18 model using transfer learning and fine-tuning.

The final model is also connected to a Gradio-based web interface where a user can upload a pet image and receive the predicted breed.

---

## Dataset

The project uses the Oxford-IIIT Pet Dataset.

Dataset source:

https://www.robots.ox.ac.uk/~vgg/data/pets/

The dataset contains:

- 7,349 images
- 37 pet breed categories
- 25 cat breeds
- 12 dog breeds
- Approximately 200 images per class
- Different image sizes, poses, lighting conditions, and backgrounds

The official dataset provides predefined train/validation and test annotations.

---

## Problem Statement

The objective is to build an image classification model that can identify the breed of a cat or dog from an input image.

### Input

An image of a cat or dog.

### Output

One of 37 possible pet breed classes.

---

## Project Workflow

The project follows these stages:

1. Data Loading
2. Data Inspection
3. Exploratory Data Analysis
4. Image Dimension Analysis
5. Class Distribution Analysis
6. Data Preprocessing
7. Data Augmentation
8. Train/Validation/Test Split
9. Custom CNN Development
10. ResNet18 Transfer Learning
11. ResNet18 Fine-Tuning
12. Model Evaluation
13. Model Comparison
14. Model Saving
15. Image Prediction
16. GUI Deployment

---

## Exploratory Data Analysis

The dataset was analyzed to understand:

- Number of images
- Number of classes
- Images per class
- Image dimensions
- Example images
- Variation between different pet breeds

The class distribution was visualized using a bar chart, and sample images were displayed to understand the visual characteristics of the dataset.

Image dimensions were also analyzed because the original dataset contains images with different sizes.

---

## Data Preprocessing

All images were converted to RGB format and resized to:

```text
224 × 224 pixels

## Model Weights

The finetuned ResNet18 weights are too large for GitHub and are hosted on Hugging Face:

**[Download oxford_pet_resnet18_finetuned.pth](https://huggingface.co/RhythmThapa/oxford-pet-resnet18/resolve/main/oxford_pet_resnet18_finetuned.pth)**

### Loading the model
```python
from huggingface_hub import hf_hub_download
import torch

model_path = hf_hub_download(
    repo_id="RhythmThapa/oxford-pet-resnet18",
    filename="oxford_pet_resnet18_finetuned.pth"
)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()
```
