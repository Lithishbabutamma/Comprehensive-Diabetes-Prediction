# 🩺 Diabetes Prediction 

This project is a complete end-to-end **Machine Learning pipeline** designed to predict whether a patient is diabetic or not based on their **clinical and demographic data**.  
It involves **data preprocessing, SMOTE balancing, exploratory data analysis (EDA), feature encoding, model comparison, and evaluation** using several popular classification algorithms.

---

## 📘 Project Overview

### 🎯 Objective
To build and evaluate multiple machine learning models on a large **clinical dataset (100,000 rows)** for predicting **diabetes occurrence**, and identify the most effective algorithm for real-world deployment.

### 💡 Key Highlights
- Data Cleaning and Preprocessing (Handling Missing Values, Encoding, Scaling)
- Exploratory Data Analysis (EDA) and Feature Correlation
- SMOTE Oversampling to Handle Class Imbalance
- Model Training & Comparison (Logistic Regression, Decision Tree, Random Forest, KNN, Naive Bayes, LightGBM, XGBoost)
- Model Evaluation with Accuracy, Recall, Precision, F1-score, ROC-AUC
- Final Model: **XGBoost with 97.16% Accuracy**
## 🧠 Dataset Information

- **Dataset Name:** Comprehensive Diabetes Clinical Dataset (100k rows)  
- **Source:** Kaggle (or internal dataset if applicable)  
- **Features:**
  - Demographic: `gender`, `age`, `location`, `race`
  - Medical: `hypertension`, `heart_disease`, `smoking_history`, `bmi`, `hbA1c_level`, `blood_glucose_level`
- **Target:** `diabetes` (0 = Non-Diabetic, 1 = Diabetic)

---

## 🔍 Exploratory Data Analysis (EDA)

- Gender, Age, Race, and Medical condition distributions  
- Correlation heatmap for numerical features  
- Outlier detection and treatment  
- Visual insights with Seaborn and Matplotlib  

📊 **Example Visualization:**
![Race Distribution](visuals/race_distribution.png)

---

## ⚙️ Model Training & Comparison

Models Used:
| Model | Accuracy | Recall (Class 1) | F1-Score | ROC-AUC |
|--------|-----------|------------------|-----------|----------|
| Logistic Regression | 0.962 | 0.67 | 0.78 | 0.85 |
| Random Forest | 0.9708 | 0.69 | 0.80 | 0.89 |
| LightGBM | 0.9715 | 0.69 | 0.80 | 0.89 |
| **XGBoost** | **0.9716** | **0.69** | **0.81** | **0.90** |
---

## 🧮 Technologies & Libraries Used

- **Language:** Python 3.12  
- **Libraries:**
  - pandas, numpy, seaborn, matplotlib  
  - scikit-learn (SMOTE, model evaluation)
  - xgboost, lightgbm  
  - imbalanced-learn, joblib  
- **IDE:** Visual Studio Code  
- **Environment:** Jupyter Notebook
 🧩 Results Summary

Best Performing Model: XGBoost

Accuracy: 97.16%

Precision: 96.7%

Recall (Diabetic class): 68.9%

F1-Score: 80.5%

AUC: 0.90

📈 The XGBoost model provides strong overall performance, maintaining a good balance between accuracy and recall for diabetic detection.
📜 License

This project is licensed under the MIT License You can use This project

👨‍💻 Author

Tamma Lithish Babu
