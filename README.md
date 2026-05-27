# Face Detection with OpenCV

A computer vision project implementing face detection using OpenCV's Haar Cascade classifiers. The project compares three different pre-trained cascade models across multiple test images to evaluate detection accuracy, false positive rates, and robustness to image variation.

## How It Works

Haar Cascade classifiers work by sliding a detection window across an image at multiple scales, applying a series of simple rectangular feature checks at each position. Features that pass all stages of the cascade are classified as faces. This approach is fast and effective for frontal face detection without requiring a GPU.

Three cascade models were tested:
- `haarcascade_frontalface_default.xml` — general purpose frontal face detector
- `haarcascade_frontalface_alt.xml` — alternative model with different feature thresholds
- `haarcascade_frontalface_alt_tree.xml` — tree-based variant with finer detection granularity

Each model was applied to the same set of test images, with detected faces marked by bounding rectangles for visual comparison.

## What Was Compared

| Model | Strengths | Weaknesses |
|---|---|---|
| default | Reliable on clear frontal faces | Can miss partially occluded faces |
| alt | Better on varied lighting | Slightly more false positives |
| alt_tree | Fine-grained detection | Slower, more sensitive to noise |

## What I Learned

This project gave me a foundation in classical computer vision before moving into deep learning-based approaches. I learned how cascade classifiers trade speed for flexibility, why they struggle with non-frontal faces and poor lighting, and how parameter tuning (scaleFactor, minNeighbors) affects the precision-recall tradeoff. It also gave me practical experience with OpenCV's image I/O and drawing utilities.

## Tech Stack

Python, OpenCV, Matplotlib, Google Colab

## Files

- `face_detection_opencv.ipynb` — full notebook with cascade classifier setup, detection function, and side-by-side comparison across all test images and models
