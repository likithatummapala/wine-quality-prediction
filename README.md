# 🍷 Red Wine Quality Prediction App

A machine learning web application to predict the quality of red wine based on its chemical properties.

This project was built as part of Data Cleaning - handling imbalanced datasets using SMOTE.

**Live Demo:https://wine-quality-prediction-k2rbcscmltaakglksukcys.streamlit.app/

### ✨ Features
- Predicts wine quality (3 to 8) from chemical inputs
- Handles imbalanced data using SMOTE (Synthetic Minority Over-sampling Technique)
- Interactive UI built with Streamlit
- Confidence score visualization

### 🛠️ Tech Stack
- Language: Python
- ML Model: RandomForestClassifier
- Libraries: Scikit-learn, Pandas, Imbalanced-learn, Streamlit
- Deployment: GitHub + Streamlit Cloud

### 📂 Dataset
- `winequality-red.csv` - Contains physicochemical properties like acidity, pH, sulphates, alcohol, etc.
- Target: quality

### 🔧 How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
