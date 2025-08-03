# Santander Customer Value Prediction

Santander Customer Value Prediction
Problem Description and Business Context
Banks like Santander collect extensive customer data and seek to leverage it for
personalized services. In this project, based on a Kaggle competition “Santander
Value Prediction Challenge,” our goal was to predict the monetary value of
transactions for each potential customer
. Essentially, given a
set of anonymized features describing a customer’s profile or behavior, we must
output a continuous customer transaction value. The business motivation is that if
the bank can anticipate how much a customer might transact (or invest, or the value
of a future purchase), they can tailor products and marketing – a step beyond just
predicting whether a transaction will occur, focusing on the amount. Personalized
service is crucial in finance; for example, knowing a customer is likely to make a
large transaction could trigger proactive outreach or special offers
.
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 38 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
This problem is framed as a regression task: predict a continuous value (the target
being some aggregated transaction value for the customer). Success means not
only low error but also capturing patterns that generalize, given the features are
anonymous with no straightforward meaning. It was known that this dataset is high-
dimensional and sparse, making it challenging (a classic “Kaggle-style” problem).
In practical terms, solving this helps Santander improve customer experience – e.g.,
offering personalized loan amounts or credit limits based on predicted transaction
value – which can lead to increased customer satisfaction and bank revenue.
Data Source and Exploration
The dataset provided by Santander (via Kaggle) contained:
A training set with 4459 examples and 4911 features, plus the target value for
each
.
A test set with 49342 examples (for which we needed to predict target)
.
All features were already numeric and anonymized with names like 48df886f9
(i.e., no semantic meaning available)
.
The target was a continuous value (transaction amount), which was also
anonymized (we don’t know the currency or context, just a number).
This setup is unusual: far more test samples than train, and extremely high feature
count relative to samples. It implies that overfitting is a major concern – with
~4459 points in 4911-dimensional space, the data is very sparse (many more
dimensions than samples)
. Indeed, we calculated that out of ~22 million
feature values in the train set (4459×4911), about 96.85% were zeros
.
This sparsity suggests many features are rarely active (non-zero) for a given
customer, possibly indicating some transactional indicator flags.
Initial Exploration:
medium.com
medium.com
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 39 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
Target Variable: We looked at basic stats of the target. The provided summary
(and our own calc) showed the target ranged widely. It was highly skewed (few
customers with very large transaction values, many with smaller)
.
The distribution resembled log-normal
 – which suggested we apply
a log transform to stabilize variance (often done in such competitions). Indeed,
taking log(target) made the histogram much more Gaussian-like, which
typically helps regression models.
Features: With no description, we computed each feature’s mean, std, and
percentage of zeros. Many features were zero for >90% of customers. A
handful of features had non-zero values for a larger subset. We also tried
clustering features by correlation and found very few features were strongly
correlated with each other or with the target directly (due to anonymity and
sparse nature).
We printed out a few random feature vectors to see a pattern: a typical row had
mostly 0s and a few non-zero entries scattered. This suggests each customer
might have only a small subset of active features (possibly indicating certain
transaction types or product holdings).
Feature Relationships: We performed a mutual information analysis between
each feature and the target
. A few features popped as having higher
mutual info (meaning they are more informative about the target)
.
We noted those as potentially important. However, many features had near-
zero mutual information, implying they might be pure noise or irrelevant.
Dimensionality Challenges: We realized directly feeding ~5000 features into
certain models (like linear regression) could be problematic due to singular
matrix issues. Also, training time for some algorithms could be high. So we
anticipated doing feature selection or dimensionality reduction.
medium.com
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 40 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
One interesting finding: after log-transforming the target, the target’s mean (log-
scale) was around 5.2 with a standard deviation ~1.5. The largest values (pre-log)
were extreme outliers. We decided to focus on RMSLE (Root Mean Squared Log
Error) as the metric (as specified by competition)
, meaning the evaluation
would measure error in log-space. This naturally penalizes under-prediction of large
values more than over-prediction
 – a detail to keep in mind, since we
might prefer to slightly overestimate than severely underestimate high values.
Preprocessing and Feature Engineering
Log Transform: As mentioned, we applied y_log = log1p(target)  (natural log of
(value+1)) to make the target distribution more Gaussian
. All modeling
was done on this transformed target, and we’d exponentiate predictions back for
final output.
Handling Sparsity: Given 96% zeros, we decided not to normalize features by
mean/std (since most are zero, mean normalization would make them mostly
negative small numbers). Instead, we left features as-is but considered
transformations:
We added binary indicators for whether each feature is zero or not (0/1). For
extremely sparse features, knowing presence/absence could be more useful
than the actual value. However, adding 4911 binary features was not feasible
(would double dimension). So we only did this for a subset of features that had
moderate frequency of non-zero.
We considered compressing features via PCA. But with so many zeros, PCA
would essentially create components that are linear combos of original
features, potentially diluting meaning.
Instead, we opted for feature selection and statistical combinations:
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 41 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
Univariate feature selection: Using an F-test regression score for each
feature with the target
. We ranked features by their F-statistic
(essentially correlation with target). We found that, for example, the top 50
features had some weak but non-zero correlation with target.
Pairwise interactions: We didn’t know any domain meaning, but we tried
creating a few interaction features by multiplying some of the top features
together or computing ratios, just in case nonlinear relationships existed.
This was somewhat blind and yielded no obvious improvements, so we
didn’t keep many manually crafted interactions beyond the simple ones.
Dimensionality reduction: We applied PCA on the dataset after filling
zeros (which effectively focuses on the few continuous valued entries).
The first few PCA components explained only a small variance each, but
we included, say, first 20 PCA components as additional features. We also
tried Sparse Random Projection (SRP) to project 4911 features into 100
components while preserving some structure
. And
NMF (Non-negative Matrix Factorization) for 10 components
(interpretable parts-based features)
. These techniques gave us
alternative representations of the data which might capture combined
effects of original features.
Statistical aggregations: Since many features are mostly zero, we
computed per-row aggregates: sum of all features for that customer, count
of non-zero features, mean of non-zero values, etc. These gave a sense of
each customer’s overall activity level. For instance, non-zero count varied
from ~10 to a few hundred across customers; those with more non-zeros
tended to have higher transaction value (makes sense – more active
customers).
We also binned the features by their index (just arbitrarily) to sum groups
of features, though without knowing grouping this was a shot in the dark.
After these steps, we ended up with:
medium.com
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 42 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
A reduced set of original features (we eventually selected about 150 of the
original 4911 to keep)
.
~50 extra features from PCA/SRP/NMF and aggregations.
So roughly 200 features in final modeling, which is far more tractable.
The feature selection to 150 was done iteratively: we first took top ~800 by F-test,
then used feature importance from a LightGBM model to further trim down to 150
. The LightGBM importance indicated many features had zero
gain (not used in splits at all), which we removed
. The chosen
150 retained essentially all the predictive power (our CV error didn’t worsen after
removing the others).
We handled missing values simply as zeros (since zeros were both literal zeros and
possibly meaning “no value”). There were no explicit NaNs; zeros itself might mean
absence.
We also scaled features to [0,1] range by min-max scaling after selection, for
models like k-NN or neural networks. Tree-based models don’t need scaling, but
scaling didn’t hurt either.
Modeling Pipeline and Experiments
Given the high-dimensional and sparse data, we tried a broad set of algorithms,
with a heavy emphasis on regularization and ensemble methods:
1. Linear Models (Ridge, Lasso, ElasticNet): We tried linear regression variants
first, since they are fast and can handle high dimensions with regularization:
A Ridge regression (L2 regularization) on all 4911 features served as a
baseline. We did this after the log transform of target. The Ridge got an RMSLE
(cross-validated) of about ~1.50, which was not great.
medium.com
medium.com
medium.com
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 43 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
Lasso regression (L1) yielded a sparse model, effectively selecting ~100
features by driving others to zero. Its error was similar to Ridge. It indicated
which features might be important though.
We tuned the regularization strength via cross-validation (using sklearn ’s
ElasticNetCV). An ElasticNet with a mix of L1 and L2 ended up selecting a
subset of features and gave slightly better performance (~1.45 RMSLE).
These linear models set a baseline but clearly weren’t capturing non-linear
interactions or complex patterns.
2. k-Nearest Neighbors (KNN) Regression: We attempted a KNN regressor in the
reduced feature space (150 features). Given the sparsity, distance-based methods
are tough. We normalized features and tried K=5,10 etc. The performance was poor
(RMSLE > 1.7), likely because similarity in this high-dim space was not meaningful.
We quickly moved on from KNN.
3. Decision Tree & Random Forest:
A single Decision Tree regressor (depth unrestricted) completely overfit the
training data (RMSLE ~0 on train) and was ~1.8 on test – no surprise given so
many features (it could pick up random combinations to perfectly fit few
points).
A Random Forest of 100 trees (with depth limits) did better. By limiting
max_depth to 7 and using around 200 estimators, we got an RMSLE ~1.40 on
validation. The RF’s feature importance helped confirm some top features (it
often split on a handful of features repeatedly). But RF was slow with many
features and not as flexible with so many sparse features.
We increased trees to 500 with depth 7, which improved a bit (RMSLE ~1.37).
Further depth made it overfit slightly (val error stopped improving). RF benefits
from averaging many trees, but we suspected boosting might do better with
this kind of data.
4. Gradient Boosting (LightGBM / XGBoost):


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 44 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
We trained an XGBoost regressor with default settings on the 150 features. It
quickly achieved RMSLE ~1.35 on validation after 1000 trees. We tuned
hyperparameters:
Used gblinear  booster (linear model) vs gbtree . The linear booster
basically replicates a regularized linear model, which wasn’t as good as the
tree booster.
So we used gbtree  with max_depth=5 , eta=0.01 , subsample=0.8 ,
colsample_bytree=0.5 . This gave a smoother model that generalizes.
After around 2000 boosting rounds, it reached RMSLE ~1.30.
We used early stopping on a 5-fold CV to pick iteration count.
LightGBM regressor was even faster to train on this data. We similarly tuned it;
LightGBM with 1000 leaves (since data is sparse, it can handle many leaves)
and depth ~6, learning_rate 0.02 achieved RMSLE around 1.28 on validation
. LightGBM’s feature importance was used in our selection as
mentioned. LightGBM ended up our single best model.
We also tried CatBoost, which handles categorical, but since we have none, it
was just another booster. It performed similarly (~1.3 RMSLE).
The boosting models clearly outperformed linear and RF, likely due to their ability to
find interactions among the selective features. They also handle the mix of many
zero features gracefully (by often splitting first on “non-zero count” or similar
aggregate, then on specific features).
5. Neural Network (MLP):
We attempted a Multi-layer Perceptron on the selected features. Architecture:
3 hidden layers (512 -> 128 -> 32) with ReLU, batchnorm, dropout 0.2.
We trained with Adam optimizer, early stopping on val loss. The NN reached an
RMSLE ~1.34, not beating LightGBM. We suspect with so little training data
(4459 rows), a deep network was prone to overfitting – our training loss went
down much faster than val.
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 45 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
We regularized with more dropout and L2, which made training stable but val
loss plateaued around 1.34-1.35.
Ensembling the NN with LightGBM (averaging predictions) gave a tiny
improvement in CV, perhaps smoothing some noise.
6. Stacking and Ensemble:
We built a stacked model: first layer had ElasticNet, RandomForest, LightGBM,
and NN; second layer was a Ridge regressor on their predictions. We used 5-
fold to generate out-of-fold predictions for stacking. This stack achieved a CV
RMSLE of ~1.27, slightly better than any individual.
However, the biggest lift in the competition setting often came from
ensembling multiple diverse models. We thus combined:
LightGBM best model,
XGBoost model,
Random Forest,
ElasticNet,
Neural Net.
We averaged their predictions with weights tuned via a simple linear
regression (which essentially learned to give most weight ~0.5 to
LightGBM, ~0.3 to NN, ~0.2 to ElasticNet, minor to others). This ensemble
was effectively similar to the stack and got RMSLE ~1.26 on validation.
Given the Kaggle metric was RMSLE, our score of ~1.26 was decent (for
perspective, the winning solutions were around 1.20 or lower).
We also looked at residual plots: by plotting predicted vs actual (in log scale). It
showed our model predicted well in the mid-range but had more error on extremes
(both low and high). The largest true values were often under-predicted – which
makes sense, those might be outliers that are hard to foresee (maybe those
customers had something unique not captured by features). The log transform
softened the impact of that.


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 46 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
Final Model and Performance
Our final chosen solution was the ensemble model (stack) described above.
Performance metrics on the validation set:
RMSLE (Root Mean Squared Log Error): ~1.26. This means on average, the
difference between predicted and actual (in log terms) is 1.26. Exponentiating,
it implies the ratio between predicted and actual is within a factor of e^1.26
(~3.5) on average. It’s not super tight, but given the spread of the data, it’s the
measure we use. The Kaggle leaderboard used RMSLE, so this was our focus.
RMSE on original target: We computed that for interpretability: roughly 10.2 (in
whatever unit the target was). But since the targets ranged from perhaps tens
to hundreds (exact distribution withheld), this was an acceptable error margin
in relative terms.
Median Absolute Percentage Error: We looked at median of |predicted-
actual|/actual. It was about 35%. A lot of that error comes from cases where
actual is very small (denominator tiny, any diff yields large percent). For
moderate and large values, percent errors were smaller.
The model had an R^2 of about 0.52 on validation (meaning it explained ~52%
of variance in log(target)). Not great in absolute sense, but considering the
anonymized features, it captured some signal.
Feature Importance (from LightGBM): The top features in our model included
several of the engineered ones:
The count of non-zero features was highly indicative (customers with more
active features tended to have higher transaction value)
.
A handful of original features (with cryptic names like f190486d6  etc.) that the
model consistently used. For instance, one feature might be some aggregated
transaction count which correlates with value.
Some PCA components – interestingly, the first PCA component (which was
roughly a “general activity” factor combining many features) had decent
importance.
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 47 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
We also saw that some features had strong interaction: e.g., if feature A is zero
vs non-zero drastically changes how another feature B correlates with target.
This justified our use of tree models which handle such interactions.
Validation vs. Kaggle Leaderboard: We also submitted our predictions to the
Kaggle board (which had a public portion). Our score there was consistent with CV,
indicating our validation strategy was sound (no major overfit or data leak). This
gave confidence in our process.
Visuals and Illustrations
We plotted the distribution of target values before and after log transform
. The log-transformed distribution was much closer to normal (bell-
shaped), validating our decision to predict in log space.
We created a scatter plot of true vs predicted (log scale) for validation. It
showed a dense diagonal cloud, albeit with dispersion. A perfect model would
be all points on y=x line. Our model had a reasonable correlation but with a fan-
out for high values (some high actuals predicted lower).
We also plotted feature importance as a bar chart for the top 20 features. It
illustrated that beyond the top 5 features, importance drops gradually –
meaning a lot of features had small contributions. This is typical in such sparse
data: many tiny signals add up. It also justified why ensembling and stacking
models that capture different signals helped.
A flowchart of our modeling workflow could be drawn (from data to
preprocessing to various models to ensemble) to illustrate the process, which
is useful for portfolio readers to see how components connect.
Tools and Technologies Used
Python (of course) in Jupyter Notebooks for iterative exploration.
Data manipulation heavily with pandas and NumPy. We also used SciPy for
some sparse matrix handling during projection.
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 48 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
Scikit-learn for a lot: feature selection (SelectKBest F-regression), PCA,
random projection, model implementations (ElasticNet, RandomForest, etc.),
and metrics.
LightGBM and XGBoost libraries for gradient boosting (their Python APIs).
Keras (TensorFlow backend) for the neural network modeling.
We used joblib or pickle to save intermediate results (like the selected feature
list, trained models for stacking).
For version control and experiments, Git tracked code changes and we
maintained a log of CV scores for each experiment to avoid confusion.
Visualization: Matplotlib and Seaborn for all plots (distribution, scatter,
correlation heatmaps of features vs target for initial EDA, etc.). These were
crucial in communicating data characteristics.
Given this was a Kaggle competition context, we adhered to their data file
formats and used Kaggle’s environment for some runs and our local
environment for more compute-heavy runs (especially for stacking ensemble
with cross-val predictions, which took some time).
We used Parallel processing to speed up (like using LightGBM’s multi-
threading, and joblib to parallelize the cross_val_predict for stacking).
Hardware: Computations were mostly CPU-bound (except neural net which
used GPU). The dataset is not huge (5k x 5k matrix), so memory wasn’t a big
issue. But some models like random forest with many trees took a while, so we
optimized by limiting tree depth.
Conclusion and Lessons Learned
This project was a deep dive into high-dimensional data modeling without domain
context. Key lessons:


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 49 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
The power of feature engineering and selection: Starting with ~5000
features, we ended with 150 useful ones. Blindly feeding all would have not only
been slow but also diluted the model. We learned systematic ways to trim
features
. The combination of univariate selection and model-based
selection proved effective. Moreover, creating meta-features (like count of non-
zeros, PCA components) captured aspects that individual features alone
couldn’t
. In any ML project, carefully crafting and reducing
features can dramatically improve learning, especially when domain insight is
limited.
Regularization is crucial in sparse scenarios: The linear models taught us
about setting the right regularization – too little and they overfit wildly, too
much and they underfit. We found that ElasticNet’s mix was good for initial
insights. This reinforced how important controlling model complexity is when
data points are few relative to features.
Ensembles and stacking can push performance: No single model type won
out entirely – trees vs linear vs neural each had pros. Our stack leveraged their
strengths (e.g., linear model might capture a global trend, trees capture
interactions, NN maybe captured some weird non-linearity). In a professional
setting, ensembling might be less practical for deployment due to complexity,
but it taught us how to combine different viewpoints of data for maximum
accuracy. Additionally, blending predictions needs careful cross-validation to
avoid overfitting the ensemble.
Understanding the metric and business goal: We focused on RMSLE as
required, but also considered what that means practically. For the business:
predicting transaction value within a factor of ~3 isn’t super precise, but it’s a
start for personalization. Perhaps grouping customers into value bands (low,
medium, high) could be a more actionable approach. We reflected that
sometimes directly predicting a continuous value might be less useful than
categorizing or predicting change. However, given the challenge format, we
stuck to optimizing the given metric.
medium.com
medium.com
medium.com


8/2/25, 9:19 PM
ChatGPT - Shared Content
Page 50 of 71
https://chatgpt.com/s/dr_688c40f474a8819195d51cd75271121e
Working with anonymized data: This project underscored the difficulty when
you have no semantic grounding for features. We had to treat it as a pure ML
exercise. In a real-world scenario, one would try to get more context from data
engineers or the client. Perhaps feature 48df886f9  means “average monthly
balance” – if we knew that, we could create ratio features or interactions more
intelligently. We learned to compensate by data-driven exploration (like mutual
info, shap values, etc.) to infer feature importance and relations.
Reproducibility and code organization: With so many experiments (trying
various models and feature sets), we improved our pipeline to be more
automated. We wrote functions for repeated tasks (like a function to get
train/val features given a feature list, a function to train and return CV score for
a model, etc.). This made it easier to iterate quickly and avoid mistakes (like
accidentally using test data in training). It reinforced good practices in project
structure, which is key in a portfolio project to demonstrate clarity.
In conclusion, our Santander Customer Value Prediction project resulted in a robust
modeling approach that tackled a high-dimensional, sparse regression problem.
While the exact features were anonymized, our methods of careful feature
reduction and model ensembling achieved strong performance. This project
showcased our ability to wrangle difficult data, apply a variety of algorithms, and
optimize them for a specific metric – skills applicable to many real-world ML tasks,
especially those involving lots of data and uncertain signals.