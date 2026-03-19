# 📊 Retail Sales Prediction using Machine Learning

## 🚀 Overview
This project is an end-to-end Machine Learning pipeline to predict **weekly retail sales** using historical data. It includes data preprocessing, feature engineering, model training, and API deployment using FastAPI.

---
## 🚀 Live demo - Deployed Link
https://retail-sales-prediction.onrender.com

## 🎯 Problem
Predict weekly sales based on factors like store, temperature, fuel price, CPI, unemployment, and date.

---

## 🧠 Approach
- Loaded and cleaned dataset using **Pandas**
- Converted date into **day, month, year**
- Selected relevant features
- Trained model using **Random Forest Regressor**
- Built API using **FastAPI** for real-time predictions

---

## ⚙️ Tech Stack
Python, Pandas, NumPy, Scikit-learn, FastAPI, Uvicorn

---

## 🔄 Workflow
Data → Cleaning → Feature Engineering → Model Training → API → Prediction

---

## ▶️ How to Run

Install dependencies:

pip install pandas scikit-learn fastapi uvicorn numpy


Train model:

python train.py


Run API:

uvicorn app:app --reload


Open:

http://127.0.0.1:8000/docs


---

## 📥 Input Example

```json
{
  "Store": 1,
  "Holiday_Flag": 0,
  "Temperature": 70,
  "Fuel_Price": 2.5,
  "CPI": 210,
  "Unemployment": 8,
  "day": 5,
  "month": 2,
  "year": 2010
}
📤 Output Example
{
  "prediction": 150000
}
