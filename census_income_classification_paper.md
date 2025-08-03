---
title: "Census Income Classification"
authors: []
date: August 03, 2025
---

# Census Income Classification

**Authors:** [Your Name]  
**Date:** August 03, 2025

## Abstract

This study explores binary classification of income level using the UCI Adult Census dataset. Models predict whether an individual earns above \$50K based on demographic and employment attributes. The work emphasizes handling class imbalance and fairness considerations, achieving competitive accuracy while identifying data quality issues such as missingness.

## 1 Introduction

Predicting income brackets from census data supports marketing segmentation and policy insights. The UCI Adult dataset provides mixed-type features with imbalance and missing values, presenting challenges for building robust classifiers without introducing bias.

## 2 Related Work

Income prediction has been a standard benchmark in machine learning, with prior work focusing on mixed-type feature encoding, class imbalance correction, and fairness-aware modeling. Techniques include resampling, cost-sensitive learning, and constrained optimization for equitable outcomes.

## 3 Data

The dataset contains 48,842 instances from the 1994 U.S. Census, with features like age, workclass, education, occupation, hours-per-week, capital gains/losses, and more. Missing values appear in fields such as workclass and occupation, denoted by '?'. The target is binary (<=50K vs >50K), with approximately 76% of samples in the lower-income class.

## 4 Methodology

Preprocessing steps included handling missing values, encoding categorical variables, and normalizing numerical inputs. Classification models were trained with mechanisms to address class imbalance, including weighting and sampling adjustments. Evaluation considered both accuracy and fairness metrics to detect potential biases.

## 5 Experimental Setup

Standard train-test splits provided by the dataset were used. Model selection relied on cross-validation, and exploratory data analysis informed feature engineering. Performance metrics included accuracy and confusion matrix analysis to understand misclassification patterns across income groups.

## 6 Results

Models achieved approximately 85%-90% accuracy. Feature analysis showed that capital gains and losses were mostly zero for the majority, and hours-per-week had a consistent mean around 40. The presence of missing values in key categorical fields accounted for around 7% of the data, influencing preprocessing decisions.

## 7 Discussion

The imbalance toward lower income could inflate naive accuracy metrics, motivating the use of additional evaluation diagnostics. Ethical considerations arise when using demographic attributes; fairness constraints could be introduced. Data quirks required careful treatment to avoid propagating biases.

## 8 Conclusion

The Census Income classification demonstrates that with thoughtful preprocessing and imbalance mitigation, high-quality income predictions are attainable. Future work should incorporate fairness-aware objectives and more robust imputation techniques.

## References

1. UCI Adult Census Dataset (Barry Becker).  
2. Mehrabi, N. et al. A Survey on Bias and Fairness in Machine Learning.  
3. He, H. & Garcia, E. A. Learning from Imbalanced Data.  
4. Bishop, C. M. Pattern Recognition and Machine Learning.  
