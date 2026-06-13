# Face Analyser

A real-time computer vision application that detects faces from a webcam feed and predicts a person's age and gender using a custom Convolutional Neural Network (CNN) trained on facial image data.

## Features

* Real-Time Face Detection using OpenCV Haar Cascades
* Age Prediction using Deep Learning
* Gender Classification
* Live Webcam Integration
* Custom CNN Architecture built with TensorFlow/Keras
* Real-Time Prediction Display on Video Feed

## Tech Stack

* Python
* Jupyter Notebook
* OpenCV
* TensorFlow / Keras
* NumPy
* Google Colab (Model Training)

## Model Details

The model is a multi-output CNN that simultaneously predicts:

* Age (Regression)
* Gender (Binary Classification)

Input Image Size:

* 128 × 128 RGB

## Project Structure

```text
Face-Analyser/
│
├── webcam_access.py
├── preprocess.py
├── train_data.py
├── compress_data.py
├── haarcascade_frontalface_default.xml
│
├── models/
│   └── face_weights.weights.h5
│
└── README.md
```

## Current Status

Active Development

Current capabilities:

* Face Detection
* Age Estimation
* Gender Recognition

Planned Improvements:

* Improved Prediction Accuracy
* Transfer Learning (MobileNetV2)
* Emotion Detection
* Better Face Tracking
* Streamlit Dashboard

## Future Enhancements

* Emotion Recognition
* Face Tracking
* Streamlit Web Interface
* Enhanced Age & Gender Accuracy
* Multi-Face Analysis