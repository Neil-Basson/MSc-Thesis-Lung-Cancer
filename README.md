# MSc_Thesis_Lung_Cancer

This repository contains the full code and model implementation for my MSc thesis:

**"Analysing the Effect of Multi-Branch Deep Neural Networks on Pulmonary Nodule Classification"**  
University of Amsterdam – MSc in Data Science and Business Analytics

---

## 🧠 Project Overview

This project develops a complete deep learning pipeline for pulmonary nodule analysis on lung CT scans. It combines:

- **Tumor segmentation** using a U-Net++ model
- **Semantic feature extraction** from binary tumor masks
- **Malignancy classification** using a multi-branch neural network (MBNN)

The goal is to evaluate how combining multiple inputs (image, mask, and features) affects classification accuracy and interpretability.

---

## 🔍 Research Contributions

- Developed a segmentation-guided, multi-branch classification model using CT scans
- Extracted 4 geometric features (compactness, solidity, diameter, spiculation proxy) from predicted masks
- Compared performance between:
  - Baseline CNN
  - Multi-branch neural network (MBNN)
- Benchmarked performance against published models using the LIDC-IDRI dataset
- Integrated Grad-CAM for model interpretability

---

## 📁 Repository Structure
MSc_Thesis_Lung_Cancer/
├── Segmentation_Model.ipynb
├── Classification_Models.ipynb
├── Process_Masks.ipynb
│
├── src/ # Source code for training and evaluation
│ ├── dataset.py
│ ├── losses.py
│ ├── mask_utils.py
│ ├── training_utils.py
│ ├── semantic_features.py
│ ├── unet_utils.py
│ ├── unet_model.py
│ └── unetpp_model.py
│
├── README.md
├── requirements.txt
└── .gitignore


---

## ⚙️ Setup & Usage

1. Clone the repo:
   ```bash
   git clone https://github.com/Neil-Basson/MSc_Thesis_Lung_Cancer.git
   cd MSc_Thesis_Lung_Cancer
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Prepare your data in the expected structure and update the paths in:
   - Process_Masks.ipynb
   - dataset.py
4. Run the segmentation model:
   - Segmentation_Model.ipynb trains U-Net++
   - Outputs binary tumor masks
5. Extract semantic features:
   - Use Process_Masks.ipynb to generate spiculation proxy, compactness, etc.
6. Train classification models:
   - Classification_Models.ipynb trains both the baseline CNN and MBNN


---
---

## 📦 Dependencies

See `requirements.txt` for a full list. Key packages include:

- `torch`
- `torchvision`
- `skimage`
- `numpy`
- `matplotlib`
- `pandas`
- `scipy`
- `tqdm`

You can install all dependencies with:

```bash
pip install -r requirements.txt
```
---


## 📊 Dataset

This project uses the **LIDC-IDRI** dataset, a publicly available collection of thoracic CT scans annotated by four board-certified radiologists. It contains over 1,000 patient scans and is commonly used in pulmonary nodule detection research.

To access the dataset, visit:  
🔗 [https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI)

### Preprocessing Steps

- CT scans are resampled and normalized.
- Lung regions are extracted using adaptive thresholding and morphological operations.
- Nodule masks are created using a majority-vote approach: pixels are included in the mask if at least **2 out of 4** radiologists annotated them.
- Tumor-centered crops of **64×64** pixels are extracted for classification.

### Dataset Splits

- **Segmentation subset:** ~9,000 annotated slices (tumor and non-tumor)
- **Classification subset:** ~2,000 nodule-centered CT crops
- Nodules with malignancy ratings:
  - **1–2** are labeled *benign*
  - **4–5** are labeled *malignant*
  - **Rating 3** is excluded due to ambiguity

---

## 🧪 Evaluation Summary

### 🔹 Segmentation (U-Net++)

| Metric               | Score   |
|----------------------|---------|
| Dice Similarity Coefficient (DSC) | 0.8846  |
| Intersection over Union (IoU)     | 0.8038  |
| Accuracy             | 0.9998  |
| Precision            | 0.8670  |

The U-Net++ model showed strong performance across all pixel-level metrics and served as the foundation for downstream classification tasks.

---

### 🔹 Classification (Ground Truth Masks as Input)

| Model     | Accuracy | AUROC | F1 Score (Macro) | False Negatives |
|-----------|----------|--------|------------------|------------------|
| CNN       | 0.8954   | 0.958  | 0.8786           | 10               |
| MBNN      | 0.9542   | 0.982  | 0.9459           | 4                |

The MBNN improved classification accuracy and significantly reduced false negatives compared to the baseline CNN, while adding minimal parameter complexity.

---

### 🔹 Classification (Predicted Masks from Segmentation Model)

| Model     | Accuracy | AUROC | F1 Score (Macro) | False Negatives |
|-----------|----------|--------|------------------|------------------|
| MBNN      | 0.9252   | 0.972  | 0.9048           | 4                |

Using segmentation-predicted masks instead of ground-truth masks resulted in a **modest drop in performance**, primarily in benign case recall. However, classification of malignant cases remained robust, with only minimal change in false negatives or AUROC.

---

## 📘 Thesis Document

This repository implements the deep learning pipeline developed for my MSc thesis:

**"Analysing the Effect of Multi-Branch Deep Neural Networks on Pulmonary Nodule Classification"**  
University of Amsterdam – MSc in Data Science and Business Analytics  
Specialization: Business Analytics  
Author: Neil Christean Basson  
Supervisor: Prof. dr. I. Birbil  
Second Reader: Prof. F. Holstege  
Date: July 10, 2025

The full thesis is included as a PDF in this repository.

---

