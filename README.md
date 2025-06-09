# 📄 Page Classification with Convolutional Networks

This project tackles the problem of **page-level classification** in scanned or digitized documents using **Convolutional Neural Networks (CNNs)**.

The model is trained to distinguish between:
- ✅ **Negative class**: Pages that should remain in the document
- ❌ **Positive class**: Pages that should be deleted (e.g., blank, separator, or irrelevant)

The classification is performed based on the **page header image**.

---

## 🚀 Features

- CNN-based binary classification model
- Focused on structural features in page headers
- Trained on labeled header crops
- Scripted pipeline for preprocessing, training, and inference
- Designed for integration into document-cleaning workflows

---

## 🧠 Model Overview

The model uses a convolutional architecture optimized for small header regions, where layout and text patterns play a key role in identifying unnecessary pages.

Input: Cropped header image of shape **[300, 1000, 1]**  
Output: Binary label (0 = keep, 1 = delete)

---

## ⚙️ Training Setup

The model is built using **TensorFlow Keras Sequential API** and trained for **binary classification**.

### 🔧 Architecture

- **Input layer**: Accepts images of shape `(300, 1000, 1)`
- **Conv2D Layer 1**: 64 filters, kernel size `(10, 20)`, strides `(3, 5)`, `ReLU`, L2 regularization
- **MaxPooling2D**: pool size `(3, 5)`
- **Conv2D Layer 2**: 128 filters, kernel size `(3, 4)`, strides `(1, 1)`, `ReLU`, L2 regularization
- **MaxPooling2D**: pool size `(2, 4)`
- **Flatten**
- **Dropout**: rate `0.20` to prevent overfitting
- **Dense Output Layer**: 1 unit with `sigmoid` activation (no bias)

### 🧪 Training Configuration

- **Optimizer**: `Adam`, learning rate `1e-3`
- **Loss Function**: `BinaryCrossentropy` (no label smoothing)
- **Metrics**:
  - `Recall` (threshold = 0.9)
  - `FalsePositives` (threshold = 0.9)

---

## 📊 Metrics

| Recall       | False Positive   | Loss         |
|--------------|------------------|--------------|
| 0.94         | 0.0              | 0.0416       |

---

## 📌 Notes

- The dataset is not included in this repository for privacy reasons.
- Evaluation is done using custom-labeled data with real-world scanned documents.