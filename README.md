# ❤️ HeartGuard — Cardiovascular Risk Prediction System

A full-stack machine learning web application that predicts cardiovascular disease risk based on health indicators, tracks user health progress over time, and provides personalized recommendations.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Flask](https://img.shields.io/badge/Flask-3.x-green) ![ML](https://img.shields.io/badge/ML-RandomForest-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🚀 Live Demo
> Coming soon — deployment in progress

---

## 📌 Project Overview

HeartGuard is not just a basic ML predictor — it's a complete health tracking platform. Users can register, take cardiovascular risk assessments, track their health progress over time, receive personalized AI-driven suggestions, and download detailed PDF health reports.

---

## ✨ Features

- **ML-Powered Risk Prediction** — Random Forest model trained on 70,000+ real patient records
- **User Authentication** — Secure registration and login with hashed passwords
- **Multi-Step Assessment Form** — Clean 3-step form with real-time input validation
- **BMI Auto-Calculation** — Automatically calculates and categorizes BMI
- **Progress Tracking** — Compares risk scores between visits and shows delta
- **Smart Personalized Suggestions** — AI-driven health insights based on what changed between visits
- **Health Dashboard** — Visual summary of health stats, trend chart, and key metrics
- **Prediction History** — Full history with risk trend chart over time
- **PDF Report Generation** — Downloadable health reports with full analysis
- **Individual Record Management** — Delete individual predictions or clear all history
- **Responsive UI** — Clean, professional light theme design

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, JavaScript, Chart.js |
| Backend | Python, Flask |
| Database | SQLite (via Flask-SQLAlchemy) |
| Authentication | Flask-Login, Werkzeug |
| Machine Learning | Scikit-learn (Random Forest, Logistic Regression) |
| PDF Generation | ReportLab |
| Dataset | Cardiovascular Disease Dataset — Kaggle (70,000 records) |

---

## 📊 Model Performance

| Metric | Logistic Regression | Random Forest |
|--------|-------------------|---------------|
| Accuracy | 72.53% | 70.63% |
| Precision | 75.25% | 70.94% |
| Recall | 66.85% | 69.56% |
| F1 Score | 70.80% | 70.24% |

> Random Forest was selected for deployment due to its superior ability to handle non-linear relationships and robustness to unseen data, despite marginally lower accuracy on the test set.

---

## 📁 Project Structure
heart-risk-predictor/
├── app.py                  # Flask application, routes, auth, DB models
├── day1_model.py           # ML training script
├── heart_model.pkl         # Trained Random Forest model
├── scaler.pkl              # StandardScaler for feature normalization
├── comparison_results.json # Model comparison metrics
├── feature_names.json      # Feature names used in training
├── heart.csv               # Dataset (Cardiovascular Disease — Kaggle)
├── requirements.txt        # Python dependencies
├── Procfile                # Deployment configuration
└── templates/
├── landing.html        # Public landing page
├── login.html          # User login
├── register.html       # User registration
├── index.html          # Multi-step prediction form
├── dashboard.html      # Health dashboard
└── history.html        # Prediction history

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/amritesh-git/heart-risk-predictor.git
cd heart-risk-predictor
```

**2. Create and activate virtual environment**
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Download the dataset**
Download the Cardiovascular Disease Dataset from [Kaggle](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and save it as `heart.csv` in the project root.

**5. Train the model**
```bash
python day1_model.py
```

**6. Run the application**
```bash
python app.py
```

**7. Open in browser**
http://127.0.0.1:5000/

---

## 🧠 ML Approach

- **Dataset:** Cardiovascular Disease Dataset (Sulianova, Kaggle) — 70,000 real patient records
- **Preprocessing:** Removed outliers in blood pressure, height, weight, and BMI. Converted age from days to years. Calculated BMI from height and weight.
- **Models Trained:** Logistic Regression and Random Forest
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1 Score
- **Selected Model:** Random Forest — better generalization on unseen data
- **Features Used:** Age, Gender, Height, Weight, BMI, Systolic BP, Diastolic BP, Cholesterol, Glucose, Smoking, Alcohol, Physical Activity

---

## 💡 What Makes This Project Unique

Most student ML projects stop at a basic prediction form. HeartGuard goes further:

1. **Longitudinal health tracking** — compares risk across multiple visits
2. **Delta-based suggestions** — personalized feedback based on what specifically changed (e.g. "You quit smoking since last visit — great progress!")
3. **Full authentication system** — each user has their own secure health data
4. **PDF report generation** — professional downloadable health reports
5. **Input validation** — real-time warnings for abnormal health values

---

## 📋 Dataset Information

- **Source:** [Cardiovascular Disease Dataset](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) by Svetlana Ulianova
- **Records:** 70,000 patient records
- **Features:** Age, Gender, Height, Weight, Blood Pressure, Cholesterol, Glucose, Smoking, Alcohol, Physical Activity
- **Target:** Presence or absence of cardiovascular disease (binary)
- **Preprocessing:** Outlier removal reduced dataset to ~68,000 clean records

---

## ⚠️ Disclaimer

This application is for informational and educational purposes only. It does not constitute medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional for medical evaluation.

---

## 👨‍💻 Developer

**Amritesh** — Information Science & Engineering, BMSIT Bengaluru  
GitHub: [@amritesh-git](https://github.com/amritesh-git)

---

## 📄 License

This project is licensed under the MIT License.