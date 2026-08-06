# Airline Passenger Satisfaction Prediction

Classification project: predicts whether a passenger is **satisfied** or
**neutral/dissatisfied**, using Logistic Regression and Random Forest.

## 1. Get the dataset
Download from Kaggle:
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

Download `train.csv` (and `test.csv` if you want — the script will combine
them automatically). Put both files in the **same folder** as
`airline_satisfaction_prediction.py`.

## 2. Set up your environment
Open a terminal in that folder and run:

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

## 3. Run the project
```bash
python airline_satisfaction_prediction.py
```

## 4. What you get
```
outputs/
├── eda/
│   ├── target_distribution.png
│   ├── satisfaction_by_class.png
│   ├── satisfaction_by_customer_type.png
│   ├── age_distribution.png
│   └── correlation_heatmap.png
├── feature_importance.png
└── model_results.txt        # accuracy, precision, recall, F1, ROC-AUC,
                              # confusion matrix, classification report
                              # for both models
```

Console output also prints the top 10 features driving satisfaction —
this is your "business insights" section.

## 5. What the script actually does (for your interview explanation)
1. **Load** — reads train/test CSVs, merges them, drops stray index columns.
2. **Clean** — fills missing `Arrival Delay in Minutes` with the median,
   drops duplicates.
3. **EDA** — target balance, satisfaction split by class/customer type/age,
   correlation heatmap.
4. **Preprocess** — label-encodes categoricals, encodes target as 0/1.
5. **Model** — 80/20 stratified split; Logistic Regression (scaled features)
   and Random Forest (200 trees, max_depth=15).
6. **Evaluate** — accuracy, precision, recall, F1, ROC-AUC, confusion matrix
   for both models, so you can compare a linear baseline vs. an ensemble.
7. **Feature importance** — Random Forest's `feature_importances_`, plotted
   and ranked, to answer "what actually drives satisfaction?"

## Typical findings on this dataset (for context)
Online boarding, in-flight wifi service, and seat comfort are usually the
top predictors — worth mentioning if your run produces similar rankings, and
worth explaining *why* if it doesn't (e.g. class imbalance in your split).

## Extending it further (optional, for a stronger portfolio piece)
- Add XGBoost/LightGBM and compare against Random Forest
- Hyperparameter tuning with GridSearchCV
- SHAP values instead of built-in feature importance (better for interviews —
  shows per-prediction reasoning, not just global ranking)
- Wrap the trained model in a small Streamlit app for a live demo
