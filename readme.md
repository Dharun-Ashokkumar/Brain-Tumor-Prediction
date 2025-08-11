# Tumour Classification using CNN (TensorFlow/Keras)

This project implements a Convolutional Neural Network (CNN) to classify brain MRI images into **three categories**:
- **Benign**
- **Malignant**
- **Normal**

The dataset is expected to be in a specific folder structure inside Google Drive, and the model is trained using **TensorFlow** with **data augmentation**.

---

## 📂 Dataset Structure

Your dataset should be organized like this inside your Google Drive:

│
├── Train/
│ ├── BENIGN/
│ ├── MALIGNANT/
│ └── NORMAL/
│
└── Test/
├── BENIGN/
├── MALIGNANT/
└── NORMAL/