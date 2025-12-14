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
