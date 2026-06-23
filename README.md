# MMA Finish Prediction Using CNN-LSTM

## Overview

This project focuses on predicting whether a Mixed Martial Arts (MMA) fight will end with a finish (KO, TKO, Submission) or continue to a decision by analyzing fight videos using deep learning techniques.

The model combines the spatial feature extraction capabilities of Convolutional Neural Networks (CNNs) with the temporal sequence learning abilities of Long Short-Term Memory (LSTM) networks, creating a robust video classification architecture capable of learning both visual and temporal fight patterns.

---

## Dataset

The dataset currently consists of:

* 140 MMA fight videos
* Finished fights (KO/TKO/Submission)
* Unfinished fights (Decision)

Kaggle Link:
https://www.kaggle.com/datasets/ouzhangn/mma-fight-ending-dataset

All videos are processed at:

* 30 FPS
* Grayscale format
* Resolution: 128 × 128 pixels

To maintain a consistent input size, 32 representative frames are sampled from each analyzed video segment.

---
### Training Preparation

The dataset is not included in this repository due to its size.

To train the model from scratch:

1. Download the dataset from Kaggle.
2. Extract the dataset to your local machine.
3. Open the preprocessing and training scripts.
4. Update all dataset paths to match your local directory structure.

For example, in `model/preprocess.py`:

```python
finished = "database/mma_finished_videos"
unfinished = "database/mma_unfinished"
```

Replace these paths with the locations where you extracted the Kaggle dataset on your system before running the preprocessing and training pipeline.

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/oguzhan-gun/mma-finish-prediction-cnn-lstm.git

cd mma-finish-prediction-cnn-lstm$
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Trained Model

Place the trained model file (`model.h5`) inside the project directory.

### 4. Run Prediction

Run the prediction script by providing a video file:

```bash
python predict.py videos/charles_vs_chandler.mp4
```

Example output:

```text
0-20s   | Prediction: 0.214
20-40s  | Prediction: 0.302
40-60s  | Prediction: 0.771

==========
FINAL RESULT: FINISHED
CONFIDENCE: 77.1%
==========
```

### Example Output

```text
0-20s   | Prediction: 0.214
20-40s  | Prediction: 0.302
40-60s  | Prediction: 0.771

==========
FINAL RESULT: FINISHED
CONFIDENCE: 77.1%
==========
```

### Supported Input

* MP4 videos
* 30 FPS recommended
* MMA fight footage
* Minimum 20 seconds duration recommended


## Video Analysis Strategy

Rather than relying on a single frame, the model analyzes temporal fight dynamics.

For inference, videos are divided into fixed 20-second windows:

```text
0 - 20 seconds
20 - 40 seconds
40 - 60 seconds
60 - 80 seconds
...
```

For each 20-second segment:

1. Frames are collected.
2. 32 frames are uniformly sampled.
3. The sampled frames are passed through the CNN-LSTM network.
4. A finish probability score is generated.

The highest confidence score among all analyzed segments is used as the final prediction.

---

## Model Architecture

The network combines:

### CNN Layers

Used for:

* Fighter posture analysis
* Distance and positioning patterns
* Ground control situations
* Visual fight dynamics

### LSTM Layers

Used for:

* Temporal action sequences
* Momentum changes
* Fight progression analysis
* Sequential pattern recognition

This CNN-LSTM combination allows the model to learn both spatial and temporal information from MMA fight footage.

---

## Performance

Current evaluation results on the test set:

| Metric | Score |
|---------|---------|
| Accuracy | 85.71% |
| Precision | 88.89% |
| Recall | 80.00% |
| F1 Score | 84.21% |
| ROC-AUC | 92.73% |

### Evaluation Visualizations

<p align="center">
  <img src="images/confusion_matrix.png" width="45%">
  <img src="images/roc_curve.png" width="45%">
</p>

---

## Future Work

The current dataset contains 140 labeled fight videos. Future development plans include:

* Expanding the dataset with additional professional MMA events
* Increasing model robustness across different promotions and camera angles
* Experimenting with larger temporal sampling strategies
* Real-time fight analysis support
* Live stream integration
* Model optimization for deployment

As more training data becomes available, further improvements in generalization and prediction accuracy are expected.

---

## Technologies

* Python
* TensorFlow / Keras
* OpenCV
* NumPy
* Scikit-Learn
* Matplotlib

---

## Author

Developed by Oğuzhan Gün

This project was developed as an independent research project focusing on deep learning, computer vision, and sports analytics.

The repository is actively maintained and future improvements, including larger datasets and enhanced model architectures, are planned.
