# 🧪 Delaney Solubility Prediction Using Machine Learning

## 📌 Project Overview
This project focuses on predicting **aqueous solubility (logS) of chemical compounds using the Delaney solubility dataset and multiple machine learning regression models.

Aqueous solubility is a critical property in drug discovery and chemical research, as it directly affects bioavailability, formulation, and therapeutic effectiveness. The objective of this project is to build, evaluate, and compare different regression models to understand how well they can predict solubility based on molecular descriptors.

This project represents complete end-to-end machine learning project, covering data preprocessing, model training, evaluation, visualization, and interpretation.

---

## 📊 Dataset Description
- **Dataset:** Delaney Solubility Dataset  
- **Target Variable:**  
  - `logS` – logarithm of aqueous solubility  
- **Features:**  
  - Molecular descriptors representing chemical and structural properties of compounds  
- **Problem Type:**  
  - Supervised regression  

The Delaney dataset is widely used in **QSAR (Quantitative Structure–Activity Relationship)** studies and serves as a benchmark dataset for solubility prediction.

---

## 🔍 Project Workflow
The project follows a structured machine learning pipeline:

### 1️⃣ Data Loading & Exploration
- Loaded the dataset using **Pandas**
- Inspected feature distributions and target variable
- Checked data structure and consistency

### 2️⃣ Feature Preparation
- Selected molecular descriptors as input features
- Split the dataset into **training** and **test** sets
- Ensured the test set remained unseen during training

### 3️⃣ Model Training
Four regression models were implemented and trained using the same data split for fair comparison:

| Model | Description |
|------|------------|
| Linear Regression | Baseline linear model |
| Random Forest Regressor | Tree-based ensemble model |
| K-Nearest Neighbors (KNN) | Distance-based regression |
| Gradient Boosting Regressor | Boosted ensemble model |

---

## 📈 Model Evaluation Strategy
Model performance was evaluated using:

- **Mean Squared Error (MSE)**  
  Measures the average squared difference between predicted and actual values.
- **R² Score**  
  Measures how well the model explains variance in the target variable.

Both **training** and **test** results were analyzed to identify:
- Overfitting
- Underfitting
- Generalization performance

---

## 📊 Results Summary

| Model | Test R² | Test MSE | Observation |
|------|--------|---------|-------------|
| Linear Regression | 0.79 | 1.02 | Strong baseline |
| Random Forest | 0.71 | 1.41 | Underfitting observed |
| KNN | 0.74 | 1.25 | Overfitting tendency |
| **Gradient Boosting** | **0.86** | **0.68** | Best overall performance |

### 🏆 Best Performing Model
The **Gradient Boosting Regressor** achieved the highest R² score and lowest MSE on the test set, indicating the best balance between bias and variance.

---

## 📉 Visualization & Model Diagnostics
To better understand model behavior and performance, the following visualizations were created:

- **Predicted vs Actual plots** for each model
- **Residual plots** to analyze prediction errors
- **Bar charts** comparing MSE and R² across models

These visual tools helped validate numerical results and diagnose overfitting and underfitting.

---

## 🎯 Key Learnings & Outcomes
Through this project, I gained practical experience in:

- Implementing multiple regression algorithms
- Comparing machine learning models using appropriate metrics
- Understanding the bias–variance tradeoff
- Detecting overfitting and underfitting
- Using visualizations for model diagnostics
- Building a complete machine learning workflow from scratch

---

## 🚀 Practical Applications
The techniques used in this project can be extended to:
- Drug discovery and pharmaceutical research
- QSAR modeling
- Chemical property prediction
- Benchmarking regression algorithms

---

## 🛠️ Tools & Technologies
- **Programming Language:** Python  
- **Libraries:**
  - NumPy
  - Pandas
  - Matplotlib
  - Scikit-learn  
- **Environment:** Jupyter Notebook / Google Collab

---

## ▶️ How to Run the Project

This project is implemented as a **Jupyter Notebook** and can be run either **locally** or **online**.

### 🔹 Option 1: Run Locally (Recommended)

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/delaney-solubility-ml.git
   
Navigate to the project folder:
cd delaney-solubility-ml

Open Jupyter Notebook:
jupyter notebook

Open the notebook file:
delaney_solubility_ml.ipynb

Run all cells from top to bottom to reproduce the results.


🔹 Option 2: Run on Google Colab (No Local Setup)
Go to https://colab.research.google.com
Click File → Open notebook
Select the GitHub tab
Paste the repository URL
Open the notebook and run all cells

📌 Requirements
The notebook uses the following Python libraries:

Python 3.x
NumPy
Pandas
Matplotlib
Scikit-learn
These libraries are commonly available in Jupyter Notebook and Google Colab environments.

🔮 Future Improvements
Potential enhancements include:
Feature scaling and feature selection
Hyperparameter tuning using GridSearchCV
Cross-validation for robust evaluation
Trying advanced models such as XGBoost or LightGBM
Deploying the model as a web application
