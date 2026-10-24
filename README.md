# Customer Churn Prediction & Retention Prioritization

Predicts how likely a telecom customer is to leave (churn) and turns that probability into a
**risk score** (High / Medium / Low) so the retention team knows who to call first.

- **Data:** Telco Customer Churn dataset (7,043 customers, 26.5% churn)
- **Model:** scikit-learn Pipeline (feature engineering → preprocessing → Logistic Regression / Random Forest)
- **App:** Streamlit web app. Enter a customer's details and get the churn risk instantly.

## How it works

1. **Features:** `src/features.py` adds 3 features: `Tenure_bucket`, `total_services`,
   `Average_monthly_spend`. It is the first step of the saved pipeline, so the model accepts raw customer data.
2. **Training:** `src/train.py` makes a stratified 80/20 split, trains Logistic Regression and Random Forest
   (both `class_weight="balanced"`), compares them on **ROC-AUC** (using `predict_proba`) and **recall**,
   and saves the best pipeline to `models/churn_pipeline.joblib`.
3. **Scoring:** `src/predict.py` converts probabilities into risk bands and saves the top-100 riskiest
   test customers to `data/processed/top100_risky.csv`.
4. **App:** `app.py` loads the model once and predicts for one customer at a time.

### Business logic (why the thresholds)

A retention offer costs **₹500**; a customer is worth **₹5,000**. If an offer saves a customer with
probability `s`, it pays off when `risk × s × 5000 > 500`, i.e. `risk > 500 / (5000 × s)`.
For `s = 25%` that is a risk of 0.40, which is why **Medium starts at 0.40** and **High is 0.70+**.
Thresholds live in `src/predict.py` (`HIGH`, `MEDIUM`) and were adjusted by checking how many
customers fall in each band.

## Project structure

```
├─ data/
│  ├─ raw/            WA_Fn-UseC_-Telco-Customer-Churn.csv
│  └─ processed/      cleaned_telco_churn.csv, featured_telco_churn.csv, top100_risky.csv
├─ models/            churn_pipeline.joblib (not in git)
├─ src/
│  ├─ features.py     add_features()
│  ├─ train.py        build, evaluate and save the pipeline
│  └─ predict.py      risk score, risk bands, top-100 list
├─ app.py             Streamlit app
├─ 01_eda.ipynb · 02_feature_engineering.ipynb · 03_model_training.ipynb
├─ requirements.txt
└─ Dockerfile
```

## How to run

```bash
# 1. Environment
python -m venv .venv
.venv\Scripts\activate            # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Data: put the raw CSV in data/raw/, then run 01_eda.ipynb once.
#    It creates data/processed/cleaned_telco_churn.csv, which train.py reads.

# 3. Train (run from the project root, with -m)
python -m src.train

# 4. Risk bands + top-100 CSV
python -m src.predict

# 5. Web app  ->  http://localhost:8501
streamlit run app.py
```

### Docker

```bash
docker build -t churn-app .
docker run -d -p 8501:8501 --name churn churn-app
# open http://localhost:8501
docker logs churn      # see errors
docker stop churn
```

Train the model first (`python -m src.train`) so `models/churn_pipeline.joblib` exists before `docker build`.

## Results

| Model | ROC-AUC | Recall |
|---|---|---|
| Logistic Regression | _fill after running_ | _fill after running_ |
| Random Forest | _fill after running_ | _fill after running_ |

Risk bands on the 1,409-customer test set (`HIGH = 0.7`, `MEDIUM = 0.4`):

| Band | Customers |
|---|---|
| High | 281 |
| Medium | 387 |
| Low | 741 |

> Note: `class_weight="balanced"` pushes scores slightly upward, so treat them as a *ranking* of risk
> more than an exact probability.