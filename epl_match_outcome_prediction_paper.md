---
title: "EPL Match Outcome Prediction"
authors: []
date: August 03, 2025
---

# EPL Match Outcome Prediction

**Authors:** [Your Name]  
**Date:** August 03, 2025

## Abstract

This study focuses on predicting English Premier League match outcomes using granular event-level data from StatsBomb. Features capturing team performance, possession, expected goals (xG), and categorical variables such as venue were engineered. Multiple models—including logistic regression, Random Forests, SVMs, and XGBoost—were trained and evaluated, revealing challenges particularly in predicting draws. Calibration and overfitting controls were applied to improve generalization.

## 1 Introduction

Predicting football match results (win, draw, or loss) has strategic applications in sports analytics, betting, and fan engagement. The English Premier League, with its rich data availability, provides an ideal testbed. This work leverages open event-level data to build and compare probabilistic classifiers for match outcomes.

## 2 Related Work

Sports outcome prediction has used various statistical and machine learning methods. Expected goals (xG) models have become standard for capturing scoring opportunity quality. Ensemble and tree-based methods have been applied alongside traditional logistic regression to improve predictive accuracy. Prior work emphasizes feature engineering around possession, shot quality, and context-aware statistics.

## 3 Data

Data were obtained from the StatsBomb open dataset in JSON format. Match metadata (home/away teams, competition context, final scores) was combined with parsed event streams to compute features such as total shots, shots on target, possession percentage, pass completion rates, xG per shot, carries into the final third, and defensive actions. Categorical features included team identity and venue.

## 4 Methodology

Feature engineering produced both continuous and categorical inputs. Models evaluated included multinomial logistic regression as a baseline, Random Forests for non-linear interactions, Support Vector Machines for margin-based classification, and XGBoost for boosted decision trees. Overfitting from team- or season-specific features was mitigated through regularization and careful feature selection.

## 5 Experimental Setup

Baseline accuracy estimates (e.g., home advantage and historical draw rates) were established. Models produced probabilistic outputs, allowing analysis of confidence and misclassification, with particular attention to the low precision in draw predictions. Evaluation used hold-out splits and cross-validation where applicable.

## 6 Results

The Random Forest model achieved balanced performance, while gradient-boosted variants (XGBoost) improved soft prediction quality. Draw outcomes remained the most difficult to predict, with precision frequently below 0.5. Trade-offs were observed between model complexity and generalization, especially when including high-dimensional team identity features.

## 7 Discussion

Calibration of predicted probabilities was essential for interpreting confidence levels. Overfitting to specific teams or seasons degraded broader applicability, suggesting the need for feature abstraction. Incorporating live data streams and adaptive reweighting could further improve responsiveness to current form.

## 8 Conclusion

Event-level data from StatsBomb, when carefully processed, supports predictive modeling of EPL match outcomes. Balancing model complexity with generalization yields more reliable predictions, and future work should include temporal updating and live integration.

## References

1. StatsBomb Open Data.  
2. Dixon, M. J. & Coles, S. G. Modelling Association Football Scores and Inefficiencies in the Football Betting Market.  
3. Chen, T. & Guestrin, C. XGBoost: A Scalable Tree Boosting System.  
4. Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.  
