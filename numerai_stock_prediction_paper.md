---
title: "Numerai Stock Prediction"
authors: []
date: August 03, 2025
---

# Numerai Stock Prediction

**Authors:** [Your Name]  
**Date:** August 03, 2025

## Abstract

This work investigates predicting stock returns using the anonymized era-based dataset provided by Numerai. Multiple machine learning models, including gradient boosting (LightGBM), logistic regression, and neural networks, were evaluated under an era-wise cross-validation framework to ensure temporal robustness. Model selection prioritized risk-adjusted consistency across market regimes. The best-performing approach balanced median correlation and variance, mitigating overfitting while delivering stable predictive signals.

## 1 Introduction

Forecasting financial returns is challenging due to non-stationarity, noise, and regime shifts. Numerai provides an encrypted, anonymized dataset where traditional domain features are hidden, forcing reliance on statistical patterns. The goal is to build models that generalize across time (eras) to predict relative stock performance, aligning incentives via staking with cryptocurrency, and enabling the ensemble to inform real-world hedge fund decisions.

## 2 Related Work

Prior research in financial machine learning has explored time-aware validation schemes, ensemble methods, and regularization to control overfitting in volatile markets. Era-based cross-validation relates to concepts in rolling-window and expanding-window evaluation. Correlation-based diagnostics and risk-adjusted metrics have been used to compare model stability across market regimes. Gradient boosting machines and neural networks are commonly employed for such high-dimensional prediction tasks.

## 3 Data

The dataset is supplied directly by Numerai and consists of anonymized numerical features per era. Each era represents a distinct time slice, and the target is a continuous value representing expected performance. The anonymization removes explicit domain signals, requiring models to extract latent structure. Era-wise splits are used for validation to simulate future performance and avoid leakage.

## 4 Methodology

Feature inputs were fed into several model families: LightGBM for gradient boosted decision trees, multinomial logistic regression for baseline calibration, and simple feed-forward neural networks implemented in Keras/TensorFlow. Hyperparameter tuning combined manual search with randomized search (RandomizedSearchCV) to balance exploration and compute efficiency. Model evaluation incorporated correlation distributions across eras and risk-adjusted performance criteria. Calibration plots and predicted vs actual diagnostics were used to examine systematic biases.

## 5 Experimental Setup

Validation employed era-aware splitting so that training and test data reflected temporal separation, reducing lookahead bias. Models were trained on past eras and evaluated on subsequent ones, with metrics aggregated to assess both central tendency and dispersion. Correlation box plots across eras provided insight into consistency, and comparisons between competing model families informed selection.

## 6 Results

Comparative analysis revealed trade-offs: some models achieved higher median correlations but exhibited larger variance across eras, while others delivered more stable yet slightly lower central performance. The preferred model offered a favorable balance, yielding reliable predictions with reduced susceptibility to regime-specific overfitting. Visualization of predicted versus actual returns suggested reasonable calibration with residual noise as expected in financial data.

## 7 Discussion

The anonymized nature of the features shifts focus from domain knowledge to robust statistical generalization. Era-wise validation was critical to avoid overfitting temporal patterns that do not persist. The model selection framework emphasized stability over raw peak performance, acknowledging that excessive variance undermines trust in production deployment. Future enhancements could include dynamic ensembling, era-specific weighting, and integrating external regime indicators.

## 8 Conclusion

This project demonstrates that with careful validation and model design, anonymized financial data can yield useful predictive models that generalize across time. Emphasizing risk-adjusted consistency leads to more deployable solutions than solely optimizing for peak metrics. Future work should explore adaptive ensembling and deeper regime-aware strategies.

## References

1. Numerai Documentation and Validation Tools.  
2. Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree.  
3. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.  
4. Chollet, F. Keras.  
5. Goodfellow, I., Bengio, Y., & Courville, A. Deep Learning.  
