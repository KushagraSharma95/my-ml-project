# Census Income Classification

## Problem Description and Business Context
Predict whether a person's income exceeds $50K annually based on demographic and employment data. Applications include customer segmentation and policy analysis, while requiring mindful handling of ethical concerns such as bias.

## Data Source and Exploration
Used UCI Adult dataset with mixed numeric and categorical features. Key characteristics:
- Imbalanced target (~24% >50K).
- Missing values in workclass, occupation, native-country.
- Strong correlations between income and features like education, marital status, occupation, hours per week.

## Preprocessing & Feature Engineering
- Handled missing values by imputation or grouping rare categories.
- One-hot encoded nominal categorical variables; label encoded binary ones.
- Created derived features: age buckets, marital indicators, capital gain/loss flags, hours-per-week tiers.
- Scaled continuous variables for linear models; tree-based methods used raw values.

## Modeling & Evaluation
- Logistic Regression: interpretable baseline with ~85% accuracy; highlighted fairness concerns (sex, marital status).
- Random Forest: improved recall for >50K class and identified important non-linear interactions.
- Gradient Boosting (XGBoost/LightGBM): top performance (~87.5% accuracy, AUC ~0.94).
- SVM and simple neural network explored; provided marginal differences.
- Ensemble approaches yielded minor gains beyond best single model.

## Metrics
- Accuracy: ~87.5%
- AUC: ~0.94
- F1 (>50K class): ~0.73
- Confusion analysis showcased trade-offs between precision and recall for high-income detection.

## Fairness Considerations
- Sensitive attributes (sex, race) contributed to disparity; explored models excluding them to quantify trade-offs.
- Highlighted importance of balancing accuracy and ethical deployment.

## Tools & Technologies
Python, pandas, scikit-learn, XGBoost, LightGBM, matplotlib, Seaborn, Jupyter.

## Conclusions and Lessons Learned
- Thorough preprocessing and feature engineering can enable simple models to perform nearly as well as complex ones.
- Awareness and mitigation of bias is critical in socio-economic modeling.
- Interpretable models (logistic) provide transparency while boosting methods offer improved accuracy.
