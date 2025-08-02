# Santander Customer Value Prediction

## Problem Description and Business Context
Predicting the monetary value of customer transactions enables proactive financial product offers and personalized service. The challenge was to model high-dimensional, sparse, anonymized customer features to estimate future transaction value, optimizing RMSLE under strong sparsity and limited labeled training data.

## Data Source and Exploration
Dataset had ~4,459 training samples with ~4,911 anonymized numerical features and a highly skewed continuous target. Extreme sparsity (~97% zeros) required careful handling. Log-transform of the target stabilized distribution for regression.

## Preprocessing & Feature Engineering
- Log transform of target (`log1p`) to reduce skew.
- Univariate feature selection (F-test) to shortlist informative features.
- Dimensionality reduction: PCA, sparse random projection, NMF components as auxiliary representations.
- Meta-features: non-zero counts, aggregate statistics, and interaction candidates.
- Iterative reduction to ~150 predictive features guided by model-based importance (LightGBM).

## Modeling and Experimentation
- Linear models (Ridge, Lasso, ElasticNet) provided baseline and sparsity understanding.
- Random Forest gave moderate improvement, but suffered overfitting when unconstrained.
- **Gradient boosting (LightGBM/XGBoost)** gave best individual performance; tuned hyperparameters for learning rate, depth, regularization.
- Neural network (MLP) used for ensembling; blended with tree models.
- Stacking of diverse learners (linear, tree, neural) using a second-layer Ridge regressor yielded best CV RMSLE (~1.27).

## Final Performance
- RMSLE: ~1.26 on validation (close to top competition scores).
- Model explained ~52% variance in log space.
- Error concentrated on extremes; mid-range predictions were most reliable.

## Tools & Technologies
Python, pandas, numpy, scikit-learn, LightGBM, XGBoost, Keras, joblib, matplotlib/seaborn, Jupyter, Git.

## Conclusions and Lessons Learned
- Systematic feature reduction and meta-feature design are essential when domain is unknown.
- Ensembling multiple perspectives (linear, tree, neural) improves robustness.
- Proper metric alignment (RMSLE) directed modeling choices and loss functions.
- Sparse, high-dimensional settings require aggressive regularization and validation discipline.
