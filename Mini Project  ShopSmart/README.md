# 🛒 ShopSmart — E-Commerce Analytics & Price Prediction

MINI PROJECT : ShopSmart Price-Predictor-ML-Pipeline   

End-To-End Data Analytics And MACHINE LEARNING PROJECT BUILT On a
SYNTHETIC E-COMMERCE DATASET Of 800 TRANSACTIONS ACROSS 7 PRODUCTS Categories.


NAME : SHREEKANTH A GUTTEDAR
---

## Project Structure

```
ShopSmart/
├── data/
│   └── shop_data.csv              ← 800 transactions (date, product, price, units, rating)
├── src/
│   ├── __init__.py
│   ├── data_loader.py             ← load, validate, clean data
│   ├── features.py                ← feature engineering (bins, date, category stats)
│   ├── visualize.py               ← 8 chart functions + dashboard
│   └── utils.py                   ← logging, save/load model, timer, JSON helpers
├── models/
│   ├── __init__.py
│   ├── train_model.py             ← RandomForest price predictor + CV + evaluation
│   ├── model.pkl                  ← saved Pipeline (auto-generated)
│   └── metrics.json               ← test/CV metrics (auto-generated)
├── reports/
│   ├── sales_analysis.md          ← static markdown report
│   ├── generate_report.py         ← auto-generate text report + charts
│   ├── sales_report.txt           ← generated text report
│   └── figures/                   ← 8 PNG charts (auto-generated)
├── notebooks/                     ← Jupyter notebooks (add your own)
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the price prediction model
python models/train_model.py

# 3. Generate the sales report (text only)
python reports/generate_report.py

# 4. Generate report + all PNG charts
python reports/generate_report.py --charts

# 5. Validate individual modules
python src/data_loader.py data/shop_data.csv
python src/features.py
python src/visualize.py
python src/utils.py
```

---

## Dataset Columns

| Column | Type | Description |
|--------|------|-------------|
| `date` | datetime | Transaction date (Jan 2024) |
| `product_id` | str | Unique product identifier |
| `product_name` | str | Product display name |
| `category` | str | One of 7 categories |
| `price` | float | Unit price (USD) |
| `units_sold` | int | Units sold in this session |
| `customer_rating` | float | 1.0 – 5.0 (nullable ~3%) |
| `sales` | float | price × units_sold |

---

## Model Performance

| Metric | Value |
|--------|-------|
| Target | `price` (regression) |
| Algorithm | RandomForestRegressor (200 trees) |
| Test MAE | ~$11 |
| Test RMSE | ~$15 |
| Test R² | **0.917** |
| CV RMSE (5-fold) | ~$18 ± $2 |

---

## Key Design Decisions

- **Pipeline wraps preprocessing** — StandardScaler and OneHotEncoder fit only on training data; zero leakage during CV.
- **Stratified train/test split** on `price_bin` preserves price-tier distribution in both sets.
- **Feature engineering in `src/features.py`** is called identically at training and inference time.
- **All functions are stateless** — every `src/` function takes a DataFrame and returns a new one; no in-place mutation.
- **`save_model` / `load_model` via joblib** — handles large NumPy arrays more efficiently than pickle.

---

## Loading the Model for Inference

```python
import joblib, pandas as pd
from src.features import add_date_features, add_price_bins, add_category_stats

pipeline = joblib.load("models/model.pkl")

# Prepare a new row (must go through the same feature engineering)
new_row = pd.DataFrame([{
    "date": pd.Timestamp("2024-02-01"),
    "product_name": "Smart Watch",
    "category": "Electronics",
    "units_sold": 3,
    "customer_rating": 4.2,
    "sales": 450.0,
}])
new_row = add_date_features(new_row)
new_row = add_price_bins(new_row)
new_row = add_category_stats(new_row)   # note: uses training-set averages

FEATURES = ["units_sold","customer_rating","cat_avg_price",
            "cat_avg_rating","cat_total_units","cat_sales_share",
            "category","price_bin"]
predicted_price = pipeline.predict(new_row[FEATURES])
print(f"Predicted price: ${predicted_price[0]:.2f}")
```
