# Zooplankton Species Classification: A Model Powered by Machine Learning

This repository contains work on using machine learning to classify zooplankton species between **Calanoid_1** and **Cyclopoid_1**. 

![image](https://github.com/user-attachments/assets/8e032961-ef9c-42f2-9fa7-e24fb12b500f)


# 🐟 Zooplankton Species Classification: A Model Powered by Machine Learning

This repository contains a machine learning project focused on classifying two morphologically similar zooplankton species—**Calanoid_1** and **Cyclopoid_1**—using high-resolution ecological datasets from **Lake Huron** and **Lake Simcoe**.

With the goal of improving efficiency and accuracy in aquatic species monitoring, we compared multiple supervised learning models and ultimately selected the most robust one through systematic evaluation.

<p align="center">
  <img src="https://github.com/yourusername/yourrepo/blob/main/images/zooplankton_comparison.png" width="400"/>
</p>

---

## 📌 Project Overview

- **Objective**: Distinguish between Calanoid_1 and Cyclopoid_1 zooplankton species using quantitative features and machine learning.
- **Datasets**:
  - `.tif` mosaic images (not used directly in model training).
  - `.csv` files containing morphological and environmental variables.
- **Lakes Used**:
  - Lake Huron (n = 51,549)
  - Lake Simcoe (n = 390,530)

---

## 🧠 Models Trained

We implemented and evaluated the following models:

- Random Forest
- Gradient Boosting
- XGBoost
- Neural Network (Fully Connected)

Each model was evaluated using:

- **F1 Score** (prioritized)
- **AUROC**
- **Accuracy**

The **Neural Network** demonstrated the most balanced and consistent performance across both lakes and was chosen as the final model.

---

## 📊 Key Results

| Lake       | Final Model    | F1 Score (%) | AUROC  | Accuracy (%) |
|------------|----------------|--------------|--------|---------------|
| Huron      | Neural Network | 96.48        | 0.9841 | 96.53         |
| Simcoe     | Neural Network | 94.06        | 0.9818 | 94.06         |

---

## 🔍 Repository Structure



