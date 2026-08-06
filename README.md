# ✈️ Airline Passenger Satisfaction Prediction

This project predicts whether an airline passenger is **Satisfied** or **Neutral/Dissatisfied** based on travel details and in-flight service ratings. It demonstrates a complete machine learning workflow, including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and feature importance analysis.

---

## 📌 Project Overview

The objective of this project is to identify the key factors influencing passenger satisfaction and build a classification model capable of predicting customer satisfaction accurately.

The project includes:

- Data Cleaning & Preprocessing
- Exploratory Data Analysis (EDA)
- Logistic Regression
- Random Forest Classifier
- Model Evaluation
- Feature Importance Analysis

---

## 📂 Dataset

The dataset used for this project is publicly available on Kaggle.

**Dataset:** https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Download `train.csv` (and optionally `test.csv`) and place them in the project directory before running the code.

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

## 📁 Project Structure

```
airline-passenger-satisfaction-prediction/
│
├── airline_satisfaction_prediction.py
├── requirements.txt
├── README.md
├── outputs/
│   ├── eda/
│   ├── feature_importance.png
│   └── model_results.txt
│
└── train.csv / test.csv
```

---

## 🚀 Installation

Clone the repository

```bash
git clone https://github.com/singhsamiksha350/airline-passenger-satisfaction-prediction.git
```

Move into the project folder

```bash
cd airline-passenger-satisfaction-prediction
```

Install the required libraries

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python airline_satisfaction_prediction.py
```

---

## ⚙️ Workflow

The project follows these steps:

1. Load and merge the dataset.
2. Handle missing values and remove duplicate records.
3. Perform exploratory data analysis (EDA).
4. Encode categorical variables.
5. Split the dataset into training and testing sets.
6. Train Logistic Regression and Random Forest models.
7. Evaluate model performance using multiple metrics.
8. Analyze feature importance to identify the most influential factors affecting passenger satisfaction.

---

## 📈 Model Performance

| Model               |   Accuracy |  Precision |     Recall |   F1 Score |    ROC-AUC |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     87.48% |     87.16% |     83.48% |     85.28% |     92.69% |
| Random Forest       | **95.83%** | **96.35%** | **93.97%** | **95.14%** | **99.29%** |

The Random Forest classifier achieved the best overall performance, significantly outperforming Logistic Regression across all evaluation metrics.

---

## 💡 Key Insights

Feature importance analysis showed that the following attributes had the greatest impact on passenger satisfaction:

- Online Boarding
- Inflight WiFi Service
- Type of Travel
- Travel Class
- Inflight Entertainment
- Seat Comfort
- Ease of Online Booking
- Leg Room Service
- Customer Type
- On-board Service

These findings highlight that both service quality and travel-related factors play a crucial role in determining customer satisfaction.

---

## 📷 Output

Running the project automatically generates:

```
outputs/
│
├── eda/
│   ├── target_distribution.png
│   ├── satisfaction_by_class.png
│   ├── satisfaction_by_customer_type.png
│   ├── age_distribution.png
│   └── correlation_heatmap.png
│
├── feature_importance.png
└── model_results.txt
```

You can also add screenshots of these outputs to this README for better visualization.

---

## 🔮 Future Improvements

- Perform hyperparameter tuning using GridSearchCV
- Compare additional ensemble models such as XGBoost or LightGBM
- Deploy the trained model using Streamlit
- Use SHAP values for model explainability

---

## 👩‍💻 Author

**Samiksha Singh**

GitHub: https://github.com/singhsamiksha350
