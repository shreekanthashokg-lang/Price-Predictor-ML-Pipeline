# 🛒 ShopSmart — E-COMMERCE ANALYTICS & PRICE PREDICTON 

### MINI PROJECT: SHOPSMART PRICE-PREDICTOR ML PIPELINE

**END-TO-END DATA ANALYTICS AND MACHINE LEARNING PROJECT**

**Author:** SHREEKANTH A GUTTEDAR
**Domain:** E-Commerce Analytics | Machine Learning | Predictive Analytics
**Project Type:** Academic / Portfolio MINI PROJECT
**Dataset:** Synthetic E-Commerce Transaction Dataset
**Records:** 800 Transactions
**Product Categories:** 7
**ML Task:** Price Prediction / Regression
**Primary Algorithm:** Random Forest Regressor

---

## 📌 PROJECT OVERVIEW

**ShopSmart** is an end-to-end E-COMMERCE analytics and machine learning PROJECT designed to analyze transaction-level shopping data and build a machine learning model capable of predicting product prices.

The project combines:

* Data loading and validation
* DATA CLEANING
* Exploratory Data Analysis (EDA)
* Feature engineering
* Statistical analysis
* Sales analytics
* Customer rating analysis
* Category-level analysis
* Machine learning model development
* Cross-validation
* Model evaluation
* Model serialization
* Automated reporting
* Data visualization

THE PROJECT USES A **synthetic DATASET containing 800 e-commerce transactions across 7 product categories**.

The MACHINE LEARNING component treats **`price` as the target variable** and uses transaction, product, category, rating, sales, and engineered statistical features to predict the expected product price.

The entire workflow is implemented as a reusable Python pipeline so that the same feature engineering and preprocessing logic can be applied during both **model training and inference**.

---

# 🎯 PROJECT OBJECTIVES

THE MAIN OBJECTIVE  of ShopSmart is to demonstrate how raw e-commerce transaction data can be transformed into actionable business insights and an operational machine learning model.

### PRIMARY OBJECTIVES 

1. Load and validate e-commerce transaction data.
2. Identify missing values and data-quality issues.
3. Clean and preprocess the dataset.
4. Analyze product and category-level sales performance.
5. Analyze pricing patterns across categories.
6. Analyze customer ratings.
7. Create meaningful derived features.
8. Build a machine learning regression pipeline.
9. Predict product prices using Random Forest Regression.
10. Evaluate model performance using multiple metrics.
11. Perform 5-fold cross-validation.
12. Save the trained model for future inference.
13. Generate automated analytical reports.
14. Produce visualization charts for business analysis.
15. Maintain a clean and modular project architecture.

---

# 💼 BUSINESS PROBLEM

E-COMMERCE platforms generate large amounts of transaction DATA containing information ABOUT:

* PRODUCTS
* CATEGORIES 
* PRICES
* UNITS SOLD
* Customer ratings
* Sales
* Transaction dates

Understanding  PRICING patterns can help businesses with:

* Pricing strategy
* Product positioning
* Category analysis
* Revenue optimization
* Inventory planning
* Product comparison
* Market analysis
* Demand-related decision making

ShopSmart demonstrates how machine learning can learn relationships between transaction characteristics and product prices.

For example:

> Given the category, units sold, customer rating, sales-related information, category statistics, and price tier, can a machine learning model estimate the expected product price?

This project answers that question using a supervised regression approach.

---

# 🔄 END-TO-END PROJECT WORKFLOW

```text
                 ┌──────────────────────┐
                 │   Raw E-Commerce     │
                 │       Dataset        │
                 │     800 Records       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Loading &       │
                 │ Validation           │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Data Cleaning        │
                 │ & Quality Checks     │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Exploratory Data     │
                 │ Analysis             │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Feature Engineering  │
                 │ Date / Category /    │
                 │ Price Features       │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Train/Test Split     │
                 │ Stratified by Price  │
                 │ Tier                 │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ ML Preprocessing     │
                 │ Scaling + Encoding   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Random Forest        │
                 │ Regressor            │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ Cross Validation     │
                 │ & Evaluation         │
                 └──────────┬───────────┘
                            │
                 ┌──────────┴───────────┐
                 ▼                      ▼
        ┌────────────────┐      ┌────────────────┐
        │ Saved ML Model │      │ Analytics      │
        │   model.pkl   │      │ Reports + PNGs │
        └────────────────┘      └────────────────┘
```

---

# 📁 PROJECT STRUCTURE

```text
ShopSmart/
│
├── data/
│   └── shop_data.csv
│       └── 800 synthetic e-commerce transactions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   │   └── Data loading, validation and cleaning
│   │
│   ├── features.py
│   │   └── Feature engineering and category statistics
│   │
│   ├── visualize.py
│   │   └── Visualization functions and dashboard generation
│   │
│   └── utils.py
│       └── Logging, model persistence, timers and JSON utilities
│
├── models/
│   ├── __init__.py
│   ├── train_model.py
│   │   └── Model training, cross-validation and evaluation
│   │
│   ├── model.pkl
│   │   └── Serialized trained ML pipeline
│   │
│   └── metrics.json
│       └── Model performance metrics
│
├── reports/
│   ├── sales_analysis.md
│   │   └── Static analytical report
│   │
│   ├── generate_report.py
│   │   └── Automated report generation
│   │
│   ├── sales_report.txt
│   │   └── Generated analytical report
│   │
│   └── figures/
│       ├── chart_01.png
│       ├── chart_02.png
│       ├── chart_03.png
│       ├── chart_04.png
│       ├── chart_05.png
│       ├── chart_06.png
│       ├── chart_07.png
│       └── chart_08.png
│
├── notebooks/
│   └── Jupyter notebooks for experimentation and EDA
│
├── requirements.txt
│
└── README.md
```

---

# 📊 DATASET DESCRIPTION

THE PROJECT uses a synthetic e-commerce transaction dataset containing **800 TRANSACTION RECORDS**.

EACH RECORD REPRESENTS a product transaction/session.

## DATASET COLUMNS 

| Column            | Data Type | Description                     |
| ----------------- | --------- | ------------------------------- |
| `date`            | datetime  | Transaction date                |
| `product_id`      | string    | Unique product identifier       |
| `product_name`    | string    | Product display name            |
| `category`        | string    | Product category                |
| `price`           | float     | Product unit price              |
| `units_sold`      | integer   | Number of units sold            |
| `customer_rating` | float     | Customer rating from 1.0 to 5.0 |
| `sales`           | float     | Total transaction sales         |

### SALES CALCULATION

```text
sales = price × units_sold
```

Example:

```text
Price = $150
Units Sold = 3

Sales = $150 × 3
      = $450
```

---

# 🏷️ PRODUCT CATEGORIES

The DATASET CONTAINS products distributed across **7 E-COMMERCE categories**.

The category information is important because products belonging to different categories can have significantly different pricing patterns.

Category-level information is therefore incorporated into the feature engineering process.

---

# 🧹 DATA PREPROCESSING

Before model training, the dataset goes through a validation and cleaning stage.

### Data validation includes:

* Checking dataset dimensions
* Checking column names
* Checking data types
* Checking missing values
* Checking duplicate records
* Checking invalid numerical values
* Checking category consistency
* Checking rating ranges
* Checking date formatting
* Verifying sales calculations

### Missing Values

`customer_rating` contains approximately **3% nullable values**.

These values are handled during preprocessing so that the machine learning pipeline can safely process the dataset.

---

# 🔎 EXPLORATORY DATA ANALYSIS

EDA is performed to understand the underlying characteristics of the e-commerce dataset.

The analysis focuses on:

### PRODUCT ANALYSIS 

* Number of unique products
* Product-level sales
* Product prices
* Units sold per product

### Category Analysis

* Category sales
* Average category price
* Total units sold by category
* Average customer rating
* Category contribution to overall sales

### CUSTOMER RATING ANALYSIS 

* Rating distribution
* Average rating
* Missing ratings
* Relationship between ratings and sales

### PRICE ANALYSIS

* Price distribution
* Price tiers
* Category-level price differences
* Relationship between price and units sold

### TIME ANALYSIS

* Transaction date distribution
* Daily transaction patterns
* Date-based feature generation

---

# 📈 DATA VISUALIZATION

The project contains **8 visualization functions** designed to communicate important business insights.

Possible analytical visualizations include:

1. Sales by Category
2. Average Price by Category
3. Units Sold by Category
4. Customer Rating Distribution
5. Price Distribution
6. Sales Distribution
7. Price vs Units Sold
8. Category Performance Dashboard

The visualizations are automatically generated into:

```text
reports/figures/
```

This makes the project suitable for both GitHub presentation and portfolio demonstration.

---

# ⚙️ FEATURE ENGINEERING

Feature engineering is one of the most important components of the project.

Raw transaction data is transformed into additional predictive variables.

The feature engineering process is implemented in:

```text
src/features.py
```

---

## 📅 DATE FEATURES

The transaction date can be transformed into useful temporal features such as:

```text
year
month
day
day_of_week
week_of_year
```

These features allow the model to capture potential time-related patterns.

---

# 💰 PRICE BINNING

Price values are divided into meaningful price tiers.

For example:

```text
Low Price
Medium Price
High Price
Premium Price
```

The resulting `price_bin` feature represents the relative pricing tier of a transaction.

This feature can help capture nonlinear pricing relationships.

---

# 🏷️ CATEGORY STATISTICS

Category-level aggregate statistics are calculated to provide additional contextual information.

The engineered features include:

```text
cat_avg_price
cat_avg_rating
cat_total_units
cat_sales_share
```

### `cat_avg_price`

Average product price within the category.

### `cat_avg_rating`

Average customer rating within the category.

### `cat_total_units`

Total number of units sold within the category.

### `cat_sales_share`

The category's contribution to total sales.

These features provide the model with broader category-level context instead of relying only on individual transaction values.

---

# 🤖 MACHINE LEARNING PROBLEM

## Problem Type

**Supervised Learning — Regression**

### Target Variable

```text
price
```

The model attempts to predict the numerical price of a product.

### INPUT FEATURES 

```text
units_sold
customer_rating
cat_avg_price
cat_avg_rating
cat_total_units
cat_sales_share
category
price_bin
```

---

# 🌲 MACHINE LEARNING ALGORITHM

The project uses:

## Random Forest Regressor

```text
RandomForestRegressor
```

with:

```text
n_estimators = 200
```

Random Forest is selected because it can effectively model:

* Nonlinear relationships
* Feature interactions
* Mixed numerical and categorical information
* Complex relationships between sales and pricing variables

It also provides a robust baseline for tabular regression problems.

---

# 🔄 MACHINE LEARNING PIPELINE

The project uses a Scikit-learn Pipeline.

Conceptually:

```text
Raw Features
     │
     ▼
Feature Selection
     │
     ▼
Numerical Preprocessing
     │
     ├── Missing Value Handling
     │
     └── StandardScaler
     │
     ▼
Categorical Preprocessing
     │
     └── OneHotEncoder
     │
     ▼
Random Forest Regressor
     │
     ▼
Predicted Price
```

The pipeline ensures that preprocessing and model training are connected into a single reusable object.

---

# 🛡️ DATA LEAKAGE PREVENTION

One of the important design decisions is preventing information from the test set from leaking into model training.

The preprocessing components are fitted only on the training data.

This includes:

* StandardScaler
* OneHotEncoder
* Missing-value handling
* Model training

The pipeline also ensures that preprocessing is performed correctly during cross-validation.

This produces a more reliable estimate of model performance.

---

# 📊 TRAIN / TEST SPLIT

The dataset is divided into training and testing sets.

A stratified split is used based on:

```text
price_bin
```

This ensures that different price tiers are reasonably represented in both training and testing datasets.

Conceptually:

```text
800 Transactions
       │
       ├───────────────┐
       │               │
       ▼               ▼
 Training Set       Test Set
       │               │
       ▼               ▼
 Model Training    Final Evaluation
```

---

# 🔁 CROSS-VALIDATION

The project uses **5-fold cross-validation**.

The training data is divided into five subsets.

```text
Fold 1 → Validation
Fold 2 → Validation
Fold 3 → Validation
Fold 4 → Validation
Fold 5 → Validation
```

Each fold is used as the validation set once while the remaining folds are used for training.

This provides a more stable estimate of model performance than relying only on a single train/test split.

---

# 📏 MODEL EVALUATION

The model is evaluated using three primary regression metrics.

## 1. Mean Absolute Error — MAE

MAE measures the average absolute difference between actual and predicted prices.

```text
MAE = Average(|Actual - Predicted|)
```

Current expected performance:

```text
Test MAE ≈ $11
```

A lower MAE indicates better prediction accuracy.

---

## 2. Root Mean Squared Error — RMSE

RMSE gives higher importance to larger prediction errors.

```text
RMSE = √Mean((Actual - Predicted)²)
```

Current expected performance:

```text
Test RMSE ≈ $15
```

Lower RMSE indicates fewer large prediction errors.

---

## 3. R² Score

R² measures how much of the variation in the target variable is explained by the model.

Current expected result:

```text
R² = 0.917
```

This means the model explains approximately **91.7% of the variance in the target price** on the test data.

---

# 🏆 MODEL PERFORMANCE

| Metric          |                Result |
| --------------- | --------------------: |
| Target          |               `price` |
| Problem         |            Regression |
| Algorithm       | RandomForestRegressor |
| Number of Trees |                   200 |
| Test MAE        |                 ≈ $11 |
| Test RMSE       |                 ≈ $15 |
| Test R²         |             **0.917** |
| 5-Fold CV RMSE  |            ≈ $18 ± $2 |

> Note: Exact metrics may vary slightly depending on the generated dataset, random state, library versions, and training configuration.

---

# 💾 MODEL SERIALIZATION

After training, the complete machine learning pipeline is saved using `joblib`.

```text
models/model.pkl
```

The saved object contains the trained preprocessing and prediction pipeline.

This allows the model to be loaded later without retraining.

Example:

```python
import joblib

pipeline = joblib.load("models/model.pkl")
```

---

# 🔮 MODEL INFERENCE

A new transaction can be passed through the same feature engineering workflow.

Example input:

```python
new_row = pd.DataFrame([{
    "date": pd.Timestamp("2024-02-01"),
    "product_name": "Smart Watch",
    "category": "Electronics",
    "units_sold": 3,
    "customer_rating": 4.2,
    "sales": 450.0,
}])
```

The same feature engineering functions are then applied:

```python
new_row = add_date_features(new_row)
new_row = add_price_bins(new_row)
new_row = add_category_stats(new_row)
```

The final features are selected:

```python
FEATURES = [
    "units_sold",
    "customer_rating",
    "cat_avg_price",
    "cat_avg_rating",
    "cat_total_units",
    "cat_sales_share",
    "category",
    "price_bin"
]
```

The prediction is generated using:

```python
predicted_price = pipeline.predict(new_row[FEATURES])
```

Finally:

```python
print(f"Predicted price: ${predicted_price[0]:.2f}")
```

---

# ⚠️ IMPORTANT INFERENCE CONSIDERATION

Category-level features such as:

```text
cat_avg_price
cat_avg_rating
cat_total_units
cat_sales_share
```

depend on category-level historical information.

Therefore, in a production environment, these statistics should be calculated from a controlled reference/training dataset or feature store rather than recalculated inconsistently for every new transaction.

This maintains consistency between training and inference.

---

# 📄 AUTOMATED REPORT GENERATION

The project contains:

```text
reports/generate_report.py
```

This script generates analytical reports automatically.

### Text-only report

```bash
python reports/generate_report.py
```

### Report + visualizations

```bash
python reports/generate_report.py --charts
```

Generated outputs include:

```text
reports/sales_report.txt
reports/figures/*.png
```

This makes the project reproducible and reduces manual reporting work.

---

# 🧩 MODULAR CODE ARCHITECTURE

The project is divided into independent Python modules.

### `data_loader.py`

Responsible for:

* Reading CSV files
* Validating columns
* Cleaning data
* Checking data quality

### `features.py`

Responsible for:

* Date features
* Price bins
* Category statistics
* Feature transformation

### `visualize.py`

Responsible for:

* Charts
* EDA visualizations
* Dashboard generation

### `utils.py`

Responsible for:

* Logging
* Timing utilities
* JSON operations
* Saving models
* Loading models

### `train_model.py`

Responsible for:

* Loading data
* Feature preparation
* Train/test split
* Pipeline creation
* Model training
* Cross-validation
* Evaluation
* Model serialization

### `generate_report.py`

Responsible for:

* Loading analytical results
* Generating text reports
* Creating visualization outputs

---

# 🚀 QUICK START

## 1. Clone the project

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd ShopSmart
```

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Train the model

```bash
python models/train_model.py
```

This generates:

```text
models/model.pkl
models/metrics.json
```

## 5. Generate the analytical report

```bash
python reports/generate_report.py
```

## 6. Generate report and charts

```bash
python reports/generate_report.py --charts
```

---

# 🧪 MODULE VALIDATION

Individual modules can also be tested.

### Data Loader

```bash
python src/data_loader.py data/shop_data.csv
```

### Feature Engineering

```bash
python src/features.py
```

### Visualization

```bash
python src/visualize.py
```

### Utilities

```bash
python src/utils.py
```

---

# 📦 REQUIREMENTS

Typical dependencies include:

```text
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
seaborn
joblib
```

Install everything using:

```bash
pip install -r requirements.txt
```

---

# 📈 EXPECTED PROJECT OUTPUTS

After successfully running the project, the following artifacts should be available:

```text
models/
├── model.pkl
└── metrics.json

reports/
├── sales_report.txt
└── figures/
    ├── chart_01.png
    ├── chart_02.png
    ├── chart_03.png
    ├── chart_04.png
    ├── chart_05.png
    ├── chart_06.png
    ├── chart_07.png
    └── chart_08.png
```

---

# 🧠 KEY DESIGN DECISIONS

## 1. Reusable ML Pipeline

Preprocessing and model training are combined into a single pipeline.

This prevents training/inference inconsistencies.

## 2. Leakage Prevention

Transformers are fitted only on appropriate training data.

This prevents test information from influencing model training.

## 3. Stratified Price Distribution

`price_bin` is used during splitting to maintain representative price tiers.

## 4. Consistent Feature Engineering

The same feature engineering logic is used during training and inference.

## 5. Stateless Functions

Functions in `src/` are designed to accept DataFrames and return transformed DataFrames without unnecessary in-place mutation.

## 6. Model Persistence

The trained pipeline is stored using `joblib`, allowing the model to be reused without retraining.

## 7. Automated Reporting

Reports and visualizations can be regenerated from the dataset through a single command.

---

# 🔐 REPRODUCIBILITY

The project is designed to be reproducible.

Important reproducibility practices include:

* Fixed random states where applicable
* Explicit feature definitions
* Modular preprocessing
* Versioned requirements
* Saved model artifacts
* Saved evaluation metrics
* Automated report generation

A new user should be able to clone the repository, install dependencies, and reproduce the main project workflow.

---

# 📊 PROJECT DELIVERABLES

The final project provides:

### Data Analytics

* Clean transaction dataset
* Product analysis
* Category analysis
* Sales analysis
* Rating analysis
* Price analysis

### Machine Learning

* Feature engineering
* Preprocessing pipeline
* Random Forest regression model
* 5-fold cross-validation
* MAE
* RMSE
* R² evaluation

### Reporting

* Markdown report
* Text report
* 8 analytical charts
* Model metrics JSON

### Deployment Readiness

* Serialized model
* Reusable inference pipeline
* Modular source code

---

# 💡 BUSINESS INSIGHTS THIS PROJECT CAN SUPPORT

The ShopSmart analytics workflow can help answer questions such as:

* Which category generates the highest sales?
* Which products have the highest average prices?
* Which categories have the highest customer ratings?
* How does price vary across product categories?
* What is the relationship between units sold and price?
* Which categories contribute most to total revenue?
* What are the dominant price tiers?
* How accurately can product prices be predicted?

---

# 🔮 FUTURE IMPROVEMENTS

The current project provides a strong foundation, but several improvements can make it closer to a production-grade system.

## Machine Learning Improvements

* Compare Random Forest with XGBoost
* Compare Random Forest with Gradient Boosting
* Hyperparameter optimization
* Bayesian optimization
* Feature importance analysis
* SHAP explainability
* Prediction confidence intervals

## Data Improvements

* Use a real-world e-commerce dataset
* Increase dataset size
* Add customer demographics
* Add product inventory
* Add discount information
* Add competitor pricing
* Add product review text
* Add historical price changes

## Analytics Improvements

* Interactive Plotly dashboard
* Power BI dashboard
* Streamlit application
* Category-level drilldowns
* Real-time sales monitoring

## Deployment Improvements

The model could be deployed through:

```text
FastAPI
      ↓
REST API
      ↓
ML Prediction Service
```

or:

```text
Streamlit
      ↓
Interactive Web Application
      ↓
Saved ML Pipeline
```

---

# 🌐 POTENTIAL PRODUCTION ARCHITECTURE

A future production version could follow:

```text
E-Commerce Database
        │
        ▼
Data Ingestion
        │
        ▼
Data Validation
        │
        ▼
Feature Engineering
        │
        ▼
Feature Store
        │
        ├───────────────┐
        ▼               ▼
Analytics          ML Training
        │               │
        ▼               ▼
Dashboard         Model Registry
                        │
                        ▼
                   Prediction API
                        │
                        ▼
                 E-Commerce System
```

---

# 🧰 TECHNOLOGY STACK

| Technology       | Purpose                   |
| ---------------- | ------------------------- |
| Python           | Core programming language |
| Pandas           | Data manipulation         |
| NumPy            | Numerical computing       |
| Scikit-learn     | Machine learning          |
| Matplotlib       | Data visualization        |
| Seaborn          | Statistical visualization |
| Joblib           | Model serialization       |
| Jupyter Notebook | EDA and experimentation   |
| Git              | Version control           |
| GitHub           | Source-code hosting       |

---

# 🎓 SKILLS DEMONSTRATED

This project demonstrates practical knowledge of:

* Python Programming
* Data Cleaning
* Data Validation
* Pandas
* NumPy
* Exploratory Data Analysis
* Statistical Analysis
* Feature Engineering
* Categorical Encoding
* Feature Scaling
* Supervised Machine Learning
* Regression
* Random Forest
* Cross-Validation
* Model Evaluation
* Data Visualization
* Modular Python Development
* Model Serialization
* Automated Reporting
* Git/GitHub Project Organization

---

# 🏆 PORTFOLIO VALUE

ShopSmart demonstrates an end-to-end workflow rather than only showing a machine learning algorithm.

The project covers the complete lifecycle:

```text
Raw Data
   ↓
Cleaning
   ↓
EDA
   ↓
Feature Engineering
   ↓
Preprocessing
   ↓
Model Training
   ↓
Cross-Validation
   ↓
Evaluation
   ↓
Model Saving
   ↓
Inference
   ↓
Automated Reporting
```

This makes it suitable as an **MCA AIML & Data Science portfolio project** because it demonstrates both data analytics and machine learning engineering practices.

---

# 📌 PROJECT SUMMARY

**ShopSmart** is a synthetic e-commerce analytics and machine learning project built around 800 transactions across 7 product categories.

The project transforms raw transaction data into meaningful analytical features and uses a **Random Forest Regression pipeline** to predict product prices.

The model currently achieves approximately:

```text
MAE  ≈ $11
RMSE ≈ $15
R²   ≈ 0.917
```

with approximately:

```text
5-Fold CV RMSE ≈ $18 ± $2
```

Beyond prediction, the project provides automated sales analysis, category-level insights, visualizations, reusable Python modules, model persistence, and reproducible reporting.

---

# 👨‍💻 AUTHOR

**Shreekanth A Guttedar**
POST GRADUATE
MCA — Artificial Intelligence & Machine Learning / Data Science

### Project Focus

```text
Data Analytics
Machine Learning
Predictive Analytics
E-Commerce Intelligence
Python
```

---

# ⭐ IF YOU FIND THIS PROJECT USEFUL

Consider giving the repository a ⭐ on GitHub.

This project is intended for **educational, academic, portfolio, and machine learning practice purposes**.
