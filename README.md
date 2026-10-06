# 📉 Customer Churn Prediction — End-to-End ML Project

Predict which telecom customers are likely to cancel, so a retention team can act early.
The project covers the full lifecycle: data → features → model comparison → tuning →
evaluation → **REST API** → **Streamlit app** → **Docker**.

## Results (held-out test set)

| Metric | Value |
|---|---|
| ROC-AUC | ~0.79 |
| PR-AUC | ~0.58 (baseline = 0.27, the churn rate) |
| Recall @ tuned threshold | ~0.72 |
| Precision @ tuned threshold | ~0.47 |

> Numbers are from the bundled **synthetic** data (`src/generate_data.py`). Swap in the real
> [IBM Telco Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) by saving it as
> `data/telco_churn.csv`; the pipeline works unchanged. Update this table with your real results.

Plots are saved to `reports/figures/`: ROC/PR curves, confusion matrix, calibration, permutation importance.

## Approach

1. **Data** – cleaning (`TotalCharges` blanks → NaN, drop ID, encode target).
2. **Features** – `avg_monthly_spend`, `charge_ratio`, `is_new_customer`, `n_addons`, then
   imputation, scaling and one-hot encoding, all inside one sklearn `Pipeline` (no train/serve skew).
3. **Model selection** – dummy baseline vs Logistic Regression vs Random Forest vs HistGradientBoosting,
   compared with stratified 5-fold CV on ROC-AUC and PR-AUC.
4. **Tuning** – `RandomizedSearchCV` on the best model.
5. **Threshold** – chosen from out-of-fold predictions to maximise F1 (the test set is never used for tuning).
6. **Evaluation** – single final pass on the held-out test set, plus calibration and permutation importance.
7. **Serving** – FastAPI (`/predict`, `/predict/batch`, `/health`) and a Streamlit UI, both using the same saved pipeline.

## Project structure

```
churn-prediction/
├── src/
│   ├── config.py         # paths & constants
│   ├── generate_data.py  # synthetic Telco-schema data generator
│   ├── data.py           # loading & cleaning
│   ├── features.py       # feature engineering + preprocessing pipeline
│   ├── train.py          # compare -> tune -> threshold -> evaluate -> save
│   ├── evaluate.py       # metrics & plots
│   └── predict.py        # inference helper
├── api/main.py           # FastAPI service
├── app/streamlit_app.py  # Streamlit demo
├── tests/                # pytest unit tests
├── notebooks/            # put your EDA notebook here
├── reports/              # metrics.json + figures/
├── Dockerfile  Makefile  requirements.txt
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data     # or place the Kaggle CSV at data/telco_churn.csv
python -m src.train             # trains, tunes, saves model + reports
pytest -q                       # run tests

uvicorn api.main:app --reload   # API docs at http://127.0.0.1:8000/docs
streamlit run app/streamlit_app.py
```

Docker: `docker build -t churn-api . && docker run -p 8000:8000 churn-api`

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"tenure": 3, "MonthlyCharges": 95.5, "TotalCharges": 286.5}'
```

## Limitations & next steps
- Synthetic data favours linear models; on real data, gradient boosting often wins, so re-check the comparison.
- Add SHAP explanations per prediction, and a cost-based threshold (cost of a retention offer vs. lost customer).
- Add drift monitoring, MLflow experiment tracking, and CI (GitHub Actions running `pytest`).
- Deploy the Streamlit app on Streamlit Community Cloud or Hugging Face Spaces and link it here.
