# EPL Match Outcome Prediction

## Problem Description and Business Context
Predicting English Premier League (EPL) match outcomes (win/draw/loss) has applications in sports analytics, betting markets, and team strategy. The task is a multi-class classification problem with class imbalance (home wins more common), requiring domain-aware feature engineering to capture tactical and form-based signals.

## Data Source and Exploration
Used StatsBomb open event data (JSON) covering multiple EPL seasons. Extracted match-level and team-level summaries from granular on-ball events:
- Aggregated features: xG, possession, pass completion, field tilt, expected threat (xT), defensive actions, and form (last 5 matches).
- Computed comparative metrics (home minus away or ratios) to reflect relative performance.
- Visualized outcome distributions and correlations: home advantage, influence of xG, and territorial control.

## Feature Engineering
- Derived high-signal metrics: expected goals (xG), possession shares, field tilt, carry and dribble threat (xT), defensive pressure indicators.
- Form-based features (momentum): recent results, goal differences over last matches.
- Target encoding approach avoided overfitting to team identity; team strength was captured through form and aggregate statistics.

## Modeling and Experimentation
### Baselines
Trivial majority class and logistic regression provided initial benchmarks (~55% accuracy with logistic).

### Advanced Models
- Random Forest improved accuracy to ~60%, better handling draws.
- SVM (linear and RBF) showed that a well-tuned linear boundary with scaled features can compete.
- XGBoost with regularization achieved ~63% accuracy after tuning class imbalance and learning rate.

### Ensembling
Stacked Random Forest, XGBoost, and logistic regression outputs with a meta-learner (logistic) to boost balanced accuracy to ~64%.

### Failure Modes & Adjustments
- Removing team ID variables to prevent overfitting to historical strengths.
- Opted not to include betting odds to maintain predictive independence.
- Neural networks and overly complex temporal aggregations yielded limited lift versus tree-based models on aggregated features.

## Final Evaluation
- **Overall accuracy:** ~64% on holdout seasons.
- **Class-wise performance:** Strong on home wins, moderate on away wins, weakest on draws.
- **ROC-AUC (one-vs-rest):** Home win ~0.80, draw ~0.67, away ~0.75.
- **Calibration check:** Predicted probabilities aligned with empirical outcome frequencies.

## Visualizations
- Confusion matrix highlighting per-class errors.
- Partial dependence plots for features like hours (domain-specific analogs) and xG differences.
- Probability calibration plots.

## Tools & Technologies
Python, pandas, numpy, scikit-learn, XGBoost, LightGBM, matplotlib/seaborn, custom StatsBomb parsers, Jupyter Notebooks, Git.

## Conclusions and Lessons Learned
- Feature engineering grounded in domain knowledge (xG, xT, field tilt) was more impactful than complex models.
- Ensembling provided marginal gains; model simplicity and interpretability were valuable.
- Overfitting to static team identity degrades generalization; comparative metrics are more robust.
- Handling class imbalance and evaluating per-class metrics is essential for fair performance interpretation.
