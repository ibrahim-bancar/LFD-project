# Deep One-Class Classification for Anomaly Detection in CIFAR-10

## Abstract
This project aims to perform anomaly detection on the **CIFAR-10** dataset using a **Deep One-Class Classification (OCC)** approach. Specifically, it leverages a **ResNet18** model in combination with the **Deep SVDD** method to learn a compact representation of the normal class (e.g., "airplane") and identify anomalies within other classes. Additionally, traditional anomaly detection methods like **Isolation Forest** and **One-Class SVM** are also used for comparison. The goal is to demonstrate that the **Deep SVDD** model outperforms traditional techniques in terms of **precision**, **recall**, and **Area Under the ROC Curve (AUC)**.

## Problem Statement
Anomaly detection is an essential task in machine learning, particularly in situations where only normal data is available. Traditional machine learning models require labeled data from both normal and anomalous classes. In contrast, **One-Class Classification (OCC)** frameworks only require data from one normal class to model anomalies. This project addresses the challenge of detecting anomalies when only normal class data is available.

## Hypothesis
It is hypothesized that combining the **Deep SVDD** algorithm with **ResNet18** for feature extraction will create a robust anomaly detection model. The model will be able to constrain normal data within a compact latent boundary and correctly identify deviations from this boundary as anomalies. It is expected that the **Deep SVDD** model will outperform traditional anomaly detection methods such as **Isolation Forest** and **One-Class SVM**.

## Methods

### Data:
The **CIFAR-10** dataset will be used, containing **60,000 color images** of 10 different classes (e.g., airplane, car, bird, etc.). For this project:
- One class (e.g., "airplane") is designated as the normal class.
- The remaining classes are considered anomalous.
- The **CIFAR-10** dataset is required for this project. You can download it using the following command:

```bash
# Download CIFAR-10 dataset
wget https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz


### Model Architecture:
1. **Feature Extraction**: We use **ResNet18** to extract high-level feature representations from the images.
2. **Anomaly Detection**: The **Deep SVDD** model is trained on the normal class features and learns to map them into a compact hypersphere. Anomalous images will fall outside this boundary.

### Comparison Methods:
- **Isolation Forest** and **One-Class SVM** will be applied to the same features extracted by ResNet18 to evaluate the performance of the Deep SVDD model.

### Expected Results:
We expect the **Deep SVDD** model to cluster normal data tightly within the hypersphere in latent space. Anomalies will deviate significantly from this boundary and should be correctly detected. This deep learning-based approach is anticipated to outperform traditional models in terms of accuracy and detection precision.

## Requirements
To run the project, install the following dependencies:

```bash
pip install -r requirements.txt
```

The **`requirements.txt`** includes:
- `torch` (PyTorch)
- `torchvision` (for ResNet18)
- `tqdm` (for progress bar)
- `scikit-learn` (for traditional anomaly detection methods)
- `numpy`
- `matplotlib`
- `pickle5` (for data serialization)




## Authors
- Ibrahim Bancar – 150220313
- Emre Aydoğmuş – 150220323

## Acknowledgments
- CIFAR-10 dataset
- **ResNet18** and **Deep SVDD** models
