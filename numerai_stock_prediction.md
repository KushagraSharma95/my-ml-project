# Numerai Stock Prediction

## Problem Description and Business Context
Numerai is a crowdsourced hedge fund tournament where participants build machine learning models to predict stock performance using encrypted feature data. The objective is to extract subtle, generalizable signals from noisy, obfuscated financial data and rank stocks according to expected future returns. Robustness and consistency matter more than raw accuracy because of the low signal-to-noise environment.

## Data Source and Exploration
Data is provided via the Numerai API and consists of weekly samples ("eras") with anonymized numeric features, an ID, and a target representing relative future return. Initial exploration confirmed:
- Features are standardized and anonymized.
- Era structure induces temporal dependencies; preserving era-level validation is critical.
- Signal is sparse; individual feature correlations are near zero, encouraging ensemble and robust validation strategies.

## Preprocessing and Feature Engineering
- Era-wise normalization and conservative neutralization to reduce exposure to any single era.
- Lightweight aggregate meta-features (e.g., per-row feature mean and std) to capture global behavior.
- Controlled feature selection guided by model importance to reduce noise while retaining distributed signal.

## Modeling Pipeline and Experiments
### Baseline
Started with logistic regression on binarized targets (top vs bottom stocks), yielding near-random performance (serving as sanity check).

### Tree-based Models
LightGBM regressor trained with era-wise cross-validation was the cornerstone, delivering stable Spearman correlations (~0.02–0.03) per era through hyperparameter tuning (shallow trees, many estimators, early stopping).

### Neural Networks
Dense neural network architectures were explored; despite tuning, they offered less stable improvements, leading to their use in ensemble only.

### Ensembling
Combined LightGBM, a neural network, and a random forest via averaging and stacking. Ensemble slightly increased consistency and reduced variance across eras.

### Validation Strategy
Custom era-wise cross-validation prevented overfitting to static validation splits. Performance was monitored both on cross-validation and live submissions.

## Final Performance and Evaluation
- **Spearman correlation:** ~0.030 average across eras.
- **Stability:** Sharpe-like consistency metric indicated low drawdowns.
- **Ranking power:** One-vs-rest AUC for separation of top vs bottom was ~0.58–0.60.
- **Interpretability:** SHAP analysis confirmed distributed influence; no single feature dominated.

## Key Visualizations
- Box plots of per-era correlations.
- SHAP summary plots showing feature contribution spread.
- Live correlation time series to monitor drift and robustness.

## Tools & Technologies
Python, Jupyter, pandas, numpy, LightGBM, scikit-learn, Keras/TensorFlow (for neural nets), SHAP, numerapi (data fetching), matplotlib/seaborn, Git for experiment tracking.

## Conclusions and Lessons Learned
- Low signal modeling demands robust validation and moderation of feature selection.
- Ensembles increased consistency more than raw predictive power.
- Treat era structure as core to any data split to avoid leakage.
- Feature importance in obfuscated domains must be interpreted cautiously; aggregated weak signals win over single strong ones.
