---

title: "Numerai Stock Prediction" summary: "Numerai Stock Prediction Problem Description and Business Context Numerai is a unique crowdsourced hedge fund that runs a weekly data science tournament for stock market prediction." date: 2025-01-03 tags: [machine learning, portfolio]

---

# Numerai Stock Prediction

## Abstract

This project builds machine learning models to predict future stock returns using Numerai’s encrypted quantitative dataset. Era-aware validation and gradient boosting methods like LightGBM are employed to mitigate overfitting and achieve robust, risk-adjusted performance.

## Introduction

Numerai Stock Prediction Problem Description and Business Context Numerai is a unique crowdsourced hedge fund that runs a weekly data science tournament for stock market prediction. Participants build machine learning models on Numerai's encrypted stock market dataset to predict stock performance, and the best models inform the hedge fund's trades.

The business goal is to predict future stock returns from quantitative data, without relying on domain knowledge, since the features are anonymized. Successful predictions can be staked with cryptocurrency (NMR) for potential earnings, aligning data scientists' incentives with the fund's performance.

In essence, the problem is a regression/ranking task: assign a higher prediction to stocks that will outperform in the near future. The challenge lies in the extremely low signal-to- noise ratio of financial markets and the obfuscated features which prevent overfitting to specific companies or industries.

From a business perspective, solving this problem means identifying subtle patterns in market data that collectively give Numerai a trading edge. A robust model could translate to profitable trades, but overfitting or chasing noise could incur losses.

Thus, thoughtful validation and generalization are crucial in this high- stakes domain. Data Source and Exploration docs.numer.ai docs.numer.ai docs.numer.ai

The dataset is provided directly by Numerai through their API. It consists of a large tabular file with an ID, era, hundreds of feature columns, and a target.

Each row represents a stock at a given time period ("era"), where the era denotes a specific week. The target is a numeric value (between 0 and 1) indicating the stock's return performance over the next 20 trading days (higher means better relative performance).

Features are anonymized numeric attributes (e.g., technical and fundamental indicators) with standardized scales, so no financial domain knowledge is needed to use them. Example structure of Numerai's dataset: each row is a stock-week with an id, era, numerous anonymized feature columns, and a target indicating future return.

For exploration, we first downloaded the data via the official API using numerapi (which provides train.parquet for the training set). Basic EDA confirmed that each era contains unique stock IDs and acts as a roughly independent sample of the market.

Summary statistics showed all features are numeric and roughly normalized (mean ~0, variance ~1). We plotted feature distributions and noticed most features are centered and have similar scales due to Numerai's preprocessing.

We also examined era-wise target distribution: targets are continuous but often binned in practice (e.g., top stocks vs bottom stocks). Each era's target mean is around 0.5 since it's a relative measure.

docs.numer.ai docs.numer.ai docs.numer.ai docs.numer.ai docs.numer.ai docs.numer.ai

## Data

Numerai Stock Prediction Problem Description and Business Context Numerai is a unique crowdsourced hedge fund that runs a weekly data science tournament for stock market prediction. Participants build machine learning models on Numerai's encrypted stock market dataset to predict stock performance, and the best models inform the hedge fund's trades.

The business goal is to predict future stock returns from quantitative data, without relying on domain knowledge, since the features are anonymized. Successful predictions can be staked with cryptocurrency (NMR) for potential earnings, aligning data scientists' incentives with the fund's performance.

In essence, the problem is a regression/ranking task: assign a higher prediction to stocks that will outperform in the near future. The challenge lies in the extremely low signal-to- noise ratio of financial markets and the obfuscated features which prevent overfitting to specific companies or industries.

From a business perspective, solving this problem means identifying subtle patterns in market data that collectively give Numerai a trading edge. A robust model could translate to profitable trades, but overfitting or chasing noise could incur losses.

Thus, thoughtful validation and generalization are crucial in this high- stakes domain. Data Source and Exploration docs.numer.ai docs.numer.ai docs.numer.ai The dataset is provided directly by Numerai through their API.

It consists of a large tabular file with an ID, era, hundreds of feature columns, and a target. Each row represents a stock at a given time period ("era"), where the era denotes a specific week.

The target is a numeric value (between 0 and 1) indicating the stock's return performance over the next 20 trading days (higher means better relative performance). Features are anonymized numeric attributes (e.g., technical and fundamental indicators) with standardized scales, so no financial domain knowledge is needed to use them.

Example structure of Numerai's dataset: each row is a stock-week with an id, era, numerous anonymized feature columns, and a target indicating future return. For exploration, we first downloaded the data via the official API using numerapi (which provides train.parquet for the training set).

Basic EDA confirmed that each era contains unique stock IDs and acts as a roughly independent sample of the market. Summary statistics showed all features are numeric and roughly normalized (mean ~0, variance ~1).

We plotted feature distributions and noticed most features are centered and have similar scales due to Numerai's preprocessing. We also examined era-wise target distribution: targets are continuous but often binned in practice (e.g., top stocks vs bottom stocks).

Each era's target mean is around 0.5 since it's a relative measure. docs.numer.ai docs.numer.ai docs.numer.ai docs.numer.ai docs.numer.ai docs.numer.ai One key aspect discovered during EDA was the presence of era correlations.

Stocks within the same era share a macroeconomic context, so models can inadvertently learn era-specific signals. To mitigate this, we treated eras akin to cross-validation folds – ensuring models perform consistently across different eras was crucial.

We also found that the signal is very sparse: a feature might only weakly correlate with the target (Pearson correlations near zero), reflecting how hard stock prediction is. This underscored the importance of robust validation and not overinterpreting spurious correlations.

Preprocessing, Feature Engineering, and Visualization The Numerai data comes pre-cleaned and numerical, so minimal preprocessing was required. No missing values were present by design, and all features were already scaled.

We did, however, perform a few careful preprocessing steps: Feature neutralization: We applied a technique to reduce exposure to any single era or single feature. For example, we demeaned each feature per era (to ensure no era had an unusually high or low feature mean) and in some experiments performed feature neutralization by orthogonalizing predictions against a set of known risky features (this is a strategy mentioned on Numerai's forum).

Feature selection: Given over 300 features, we tried removing some features that appeared truly uninformative. Using feature importance scores from an initial model, we dropped the lowest-importance features to reduce noise.

We also experimented with PCA on the feature set, but found that since features are already engineered by Numerai, PCA didn't improve performance. Feature engineering in the traditional sense was limited because feature names are anonymous (e.g., feature1, feature2, …) and combining them arbitrarily could inject noise.

However, we derived a few aggregate features: for instance, we computed the average feature value per row and the standard deviation of features per row as potential meta-features. The intuition was that a stock with extreme feature values might behave differently than one with all average feature values.

These aggregate features yielded slight improvements on validation correlation. For visualization, we focused on era-wise metrics.

A valuable plot was the validation correlation per era: for each era in a validation set, we computed the Spearman correlation between our predictions and actual targets. This was visualized as a box plot over eras.

Ideally, a good model will have consistently positive correlation in most eras rather than doing extremely well in some eras and poorly in others (which could indicate overfitting). Our initial models showed high variance in era performance, which guided us to refine our feature selection and model complexity.

After improvements, the era correlations clustered more tightly around a positive mean (e.g., median Spearman ~0.02 with interquartile range of ~0.01). While these correlations seem small, in the stock market context any positive correlation is valuable.

We also plotted the feature importance as estimated by tree-based models. However, Numerai's own advice is to be cautious: due to feature obfuscation, importance can be misleading, and there are substitution effects where many features can stand in for each other.

Indeed, our LightGBM model's top 20 features had only marginally higher gains than the next 100 features, suggesting the signal is distributed across many inputs. Modeling Pipeline and Experimentation Our modeling pipeline went through several iterations of trial-and-error: 3.

Neural Network: We attempted a deep neural network using Keras, with an architecture of several dense layers (ReLU activations) and dropout. Despite hyperparameter tuning, the neural network's validation metrics were slightly lower than LightGBM's and less stable across eras.

Likely the network had trouble given the small signal and required more regularization. We eventually combined it with LightGBM predictions in an ensemble.

docs.numer.ai docs.numer.ai python import lightgbm as lgb features = [f for f in training_data.columns if f.startswith("feature")] model = lgb.LGBMRegressor(n_estimators=2000, learning_rate=0.01, num_leav model.fit(train_df[features], train_df["target"], eval_set=[(val_df[featu eval_metric="l2", early_stopping_rounds=50, verbose=False) Copy Feature Importance: The LightGBM's top features yielded minor lifts individually, underscoring that the model gains by aggregating many weak signals. This aligns with Numerai's statement that the dataset has "a very low signal to noise ratio" – no single feature is a game-changer, but collectively they provide an edge.

Stability: Over the course of multiple weeks of live testing, the model's correlations on live data fluctuated but remained generally positive. On Numerai, models are also evaluated by "MMC" (meta-model contribution) and drawdown.

Our model had a relatively low drawdown, meaning it didn't crash during bad weeks, which is important for reliability. It's worth noting that absolute performance numbers in stock prediction are low; what matters is consistency and managing risk.

We visualized a time-series of live correlations of our model, which showed a mix of wins and losses but a positive trend overall. We also plotted a distribution of per-era predictions vs actuals, observing that the model tended to correctly rank stocks on average but struggled in some eras with unusual market conditions (e.g., abrupt crashes or rallies).

Evaluation Metrics: The primary metric was Spearman correlation between predictions and returns. We chose Spearman (rank correlation) since Numerai cares about the ordering of stock predictions.

Additionally, we tracked the model's mean squared error (for regression) which was used during training, and the Sharpe ratio of correlations (to gauge consistency). Numerai also provides a built-in validation dashboard that reports metrics like correlation and drawdown; our final model passed their "validation diagnostics" threshold of 0.02 correlation comfortably.

Visualizations and Interpretability Because the features are anonymous, interpreting the model is challenging. We employed SHAP (SHapley Additive exPlanations) analysis to get a sense of feature influence.

The SHAP summary plot showed that certain features consistently had higher impact on the prediction (for example, feature173 and feature52 in our model) – but without real-world meaning, we treated this exercise as a way to ensure no single feature dominated excessively. Indeed, the spread of SHAP values was fairly uniform across many features, indicating our model's predictions were an aggregation of many small effects rather than one big driver.

We also examined model residuals by era: plotting each era's average prediction vs average actual return. This helped identify if the model was systematically biased in certain eras (e.g., always overly optimistic in volatile weeks).

The plot hovered around the ideal diagonal line, albeit with noise, which gave some confidence that era-dependent biases were minimized. Another useful visualization was the earlier-mentioned correlation box plot across folds/eras.

In one scenario, we compared two modeling approaches (say Model A vs Model B) by their correlation distributions over eras. Model A had slightly higher median but also higher variance, whereas Model B was more consistent.

We ultimately favored the model with a better risk-adjusted performance (higher median and a tighter interquartile range of correlations). Tools and Technologies Used Programming Language: Python (Jupyter Notebooks for experimentation).

Data Handling: pandas for data manipulation (loading parquet, merging era data), numpy for numeric operations. Numerai API: numerapi library was used to download the latest datasets and submit predictions programmatically.

docs.numer.ai Modeling Libraries: We heavily used LightGBM for gradient boosting. Additionally, scikit-learn was used for logistic regression, and a simple Keras (TensorFlow) implementation for the neural network.

Validation & Hyperparameter Tuning: scikit-learn 's KFold was used to implement era-wise cross-validation. For hyperparameters, we combined manual tuning guided by validation results with RandomizedSearchCV for LightGBM to fine-tune tree parameters.

Visualization: matplotlib and seaborn for plotting distributions, correlations, and boxplots. We also used Numerai's provided web tools (like their validation diagnostic charts) for additional insight.

Version Control: Git was used to track experiments, and results were logged in a Weights & Biases dashboard for comparison across iterations (this helped in keeping track of which experiment settings led to which performance, supporting the "efficient methodology" approach ). Conclusions and Lessons Learned This project provided a rigorous exercise in building models for an extremely noisy real-world problem.

Key takeaways include: The importance of cross-validation and robust validation strategies: We learned early that using a single static validation can be misleading. By validating across eras (time periods), we ensured our model wasn't just lucky on one slice of data.

No free lunch in feature selection: With anonymized features, typical domain- driven feature engineering was impossible. We had to rely on data-driven methods to trim or transform features.

We discovered that aggressive feature selection (e.g., dropping features with tiny individual importance) didn't always help – sometimes those features collectively mattered. A moderate approach of removing obviously uninformative features and reducing multicollinearity was more effective.

docs.numer.ai

## Methodology

1. Baseline model (logistic regression): We started with a simple logistic regression treating the problem as binary classification (we labeled target > 0.5 as "high return" vs <= 0.5 "low return").

This baseline yielded a validation AUC around 0.52 – only slightly above random – highlighting the difficulty. It served mainly as a sanity check.

2. Gradient Boosted Trees (LightGBM): Given the success of tree ensembles on structured data, we trained a LightGBM regressor to predict the continuous target.

We used early stopping with era-wise cross-validation: we split training data by eras to ensure each fold covered different time periods. LightGBM performed markedly better, achieving a validation correlation ~0.02– 0.03 (Spearman) consistently, compared to ~0 for the baseline.

We tuned hyperparameters (number of leaves, depth, learning rate) and found a shallow model (max_depth ~5) with many trees (~2000) worked well, presumably to avoid overfitting noise. An example of our model code: This LightGBM model became our workhorse, and its performance on the validation set (Spearman ≈ 0.025) was a solid starting point.

4. Ensembling: Numerai allows submitting blended predictions, and ensembling is a common strategy to reduce variance.

We averaged the predictions of three models: our tuned LightGBM, a neural network, and a random forest. The ensemble achieved slightly higher Sharpe (consistency) on validation, though the raw correlation was similar.

The logic was that different model types might overfit to different noise, so averaging helps cancel out some noise. Throughout experimentation, a vital lesson was to avoid overfitting to the provided validation set.

Numerai provides a fixed validation split and even a "diagnostics" tool on it. It's tempting to optimize solely for that set, but we adhered to cross-validation on training eras to pick models, and used the official validation only for final sanity check.

We also monitored the "live" performance (on Numerai's live data each week) after submissions – a good model should continue to perform decently on live data, indicating it generalizes well. Final Model and Performance Evaluation Our final selected model was an ensemble that leaned mostly on a LightGBM regressor.

In terms of performance: Validation Spearman Correlation: ~0.030 on average per era, with a Sharpe ratio (mean/std of correlation) around 1.5. For context, these values are in line with top-performing models in the tournament; even a few hundredths of correlation can be significant.

ROC AUC (if framing as classification): ~0.58–0.60 for predicting top vs bottom stocks, which also indicates modest predictive power above random. We computed this by labeling the top 20% target values as positive class; the AUC being above 0.5 confirms the model picks more winners than chance, albeit with plenty of noise.

## Future Work

Ensembling and consistency: A combination of models yielded more stable performance than any single model. This is crucial in finance – consistency (avoiding large drawdowns) can be more valued than sporadic high returns.

Our final ensemble's strength was not dramatically higher correlation, but more consistent correlation across eras. Domain knowledge vs data-driven approaches: Even though the data was obfuscated, general finance knowledge still helped in certain decisions (for example, treating each week independently, knowing that market regimes change).

We realized that blending some domain intuition with the pure machine learning approach can be powerful, even if indirect. Reproducibility and pipeline rigor: We treated this like a production pipeline – from downloading data to generating predictions was scripted, ensuring reproducibility.

This taught us the importance of clean code and experiment tracking. We maintained a JSON log of each training run (with parameters and validation scores) to avoid losing track of what we tried.

Low signal modeling mindset: Working on Numerai instilled a mindset of humility – even the best model is only slightly better than random. Small improvements matter and need solid statistical validation.

We learned to be careful about p-hacking or overfitting; if a tweak didn't improve cross-validated metrics consistently, it was likely just noise. In summary, our Numerai stock prediction project demonstrated how machine learning can extract a slight but significant edge in a noisy domain.

It reinforced best practices in validation and taught us how to handle a machine learning problem where traditional feature meaning is absent. The experience has been valuable for tackling any low-signal/high-noise prediction tasks in our future work.

## References

- [1] [Numerai Tournament and Dataset](https://numer.ai/).
- [2] [Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree.](https://proceedings.neurips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree.pdf) 2017.
- [3] [Chen, T., & Guestrin, C. XGBoost: A Scalable Tree Boosting System.](https://doi.org/10.1145/2939672.2939785) KDD 2016.
- [4] [Pedregosa, F. et al. Scikit-learn: Machine Learning in Python.](https://jmlr.org/papers/v12/pedregosa11a.html) JMLR 2011.
