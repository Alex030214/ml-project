# Boston Airbnb Price Prediction — DATA1030 Final Project

**Author:** Zhuolei Chen  
**Affiliation:** Data Science Institute, Brown University

## Project overview
Supervised regression to predict nightly listing price (USD) for Boston Airbnb.  
Pipeline includes: EDA, preprocessing with proper imputation and encoding, fixed train/val/test split, cross-validation with uncertainty, baseline comparison, model selection, and interpretability (global importances + SHAP). All figures are saved at 300 dpi.

## Reproducibility
This project was developed and tested with the following exact versions:

- **Python**: 3.13.5
- **NumPy**: 2.1.3
- **Pandas**: 2.2.3
- **Scikit-learn**: 1.6.1
- **Matplotlib**: 3.10.0
- **SHAP**: 0.47.2

A complete Conda spec is provided in **`environment.yml`**.

### Setup (Conda)
```bash
conda env create -f environment.yml
conda activate ml-project
python -c "import sklearn, shap; print('sklearn', sklearn.__version__, 'shap', shap.__version__)"
```

### Run order

Open Jupyter in the repo root and execute notebooks in order:
```
src/notebooks/01_quick_eda.ipynb

src/notebooks/02_preprocess_split.ipynb

src/notebooks/03_models_cv.ipynb

src/notebooks/04_feature_importance_and_shap.ipynb

src/notebooks/05_error_analysis.ipynb
```
Outputs:

Figures (300 dpi): figures/

Arrays, tables, and metrics: results/

### Data

Target: price

Features used:
Categorical — room_type, neighbourhood_cleansed
Numeric — latitude, longitude, minimum_nights, number_of_reviews, reviews_per_month, availability_365, calculated_host_listings_count
Engineered — dist_to_center_km (Haversine distance to downtown Boston)

Place the CSVs in data/ (already included in this repo). If using an external source, reference the Boston 2016 snapshot and keep the same columns.

### Methods summary

Preprocessing:

Numeric: KNNImputer(n_neighbors=5, weights="distance") + MissingIndicator + StandardScaler

Categorical: OneHotEncoder(handle_unknown="ignore", sparse_output=False) (NaN kept as its own category)

Split: fixed 60/20/20 (train/val/test)

Models: Linear Regression, Ridge, Lasso, KNN, Random Forest (max_depth ∈ {4,6,8,10}), Gradient Boosting

Evaluation: MAE, RMSE, R²; 5-fold CV with multiple seeds; baseline = median price

Interpretability: Gini, permutation, drop-column importances; SHAP (beeswarm, bar, waterfall)

### Repository layout
```
.
├── data/              # CSVs or data link
├── figures/           # 300 dpi figures
├── results/           # saved arrays, tables, metrics
├── report/            # final PDF goes here
├── src/
│   └── notebooks/     # 01~05 notebooks (run in order)
├── environment.yml
├── LICENSE
└── README.md
```

### License

MIT License (see LICENSE).


---
