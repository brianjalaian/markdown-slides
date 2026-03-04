# Combining Different Models for Ensemble Learning
### CAP6606: Machine Learning for ISR
#### Dr. Brian Jalaian
##### Based on Chapter 7, Raschka & Mirjalili

<div style="text-align: right"><font size="4">1</font></div>

---
### Chapter Overview

- Learning with ensembles
- Combining classifiers via majority vote
- Bagging: Bootstrap aggregating
- Leveraging weak learners via adaptive boosting
- Gradient boosting and XGBoost

<div style="text-align: right"><font size="4">2</font></div>

---
### Learning with Ensembles

**Goal:** Combine multiple classifiers to build a more robust model

**Key Insight:** Individual classifiers may make errors, but if errors are uncorrelated, the ensemble can outperform any single model

**Mathematical Foundation:** Ensemble error decreases when base classifiers are better than random guessing

<div style="text-align: right"><font size="4">3</font></div>

---
### Ensemble Error Probability

For $n$ classifiers with error rate $\epsilon$, the probability that the majority is wrong:

$$P(\text{error}) = \sum_{k > n/2}^{n} \binom{n}{k} \epsilon^k (1-\epsilon)^{n-k}$$

```python
from scipy.special import comb
import math

def ensemble_error(n_classifier, error):
    k_start = int(math.ceil(n_classifier / 2.))
    probs = [comb(n_classifier, k) *
             error**k * (1-error)**(n_classifier - k)
             for k in range(k_start, n_classifier + 1)]
    return sum(probs)
```

<div style="text-align: right"><font size="4">4</font></div>

---
### Ensemble Error vs Base Error

```python
ensemble_error(n_classifier=11, error=0.25)
# Output: 0.034
```

| Base Error | Ensemble Error (n=11) |
|:----------:|:---------------------:|
| 0.25 | 0.034 |
| 0.35 | 0.133 |
| 0.45 | 0.417 |
| 0.50 | 0.500 |

**Key:** Ensembles only help when base error < 0.5

<div style="text-align: right"><font size="4">5</font></div>

---
### Combining Classifiers via Majority Vote

**Two Voting Strategies:**

- **Hard Voting (Majority Vote):** Predict the class label that received the most votes

- **Soft Voting:** Predict based on the highest average probability across classifiers

```python
# Hard voting example
np.argmax(np.bincount([0, 0, 1],
                      weights=[0.2, 0.2, 0.6]))
```

<div style="text-align: right"><font size="4">6</font></div>

---
### Soft Voting Example

```python
# Probabilities from 3 classifiers
ex = np.array([[0.9, 0.1],   # Classifier 1
               [0.8, 0.2],   # Classifier 2
               [0.4, 0.6]])  # Classifier 3

# Weighted average probabilities
p = np.average(ex, axis=0,
               weights=[0.2, 0.2, 0.6])
# p = [0.58, 0.42]

np.argmax(p)  # Predicts class 0
```

<div style="text-align: right"><font size="4">7</font></div>

---
### MajorityVoteClassifier Implementation

```python
class MajorityVoteClassifier(BaseEstimator,
                             ClassifierMixin):
    def __init__(self, classifiers,
                 vote='classlabel', weights=None):
        self.classifiers = classifiers
        self.vote = vote
        self.weights = weights

    def fit(self, X, y):
        self.classifiers_ = []
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X, y)
            self.classifiers_.append(fitted_clf)
        return self
```

<div style="text-align: right"><font size="4">8</font></div>

---
### Majority Vote: Prediction Methods

```python
def predict(self, X):
    if self.vote == 'probability':
        maj_vote = np.argmax(self.predict_proba(X),
                             axis=1)
    else:  # 'classlabel' vote
        predictions = np.asarray(
            [clf.predict(X) for clf in self.classifiers_]
        ).T
        maj_vote = np.apply_along_axis(
            lambda x: np.argmax(
                np.bincount(x, weights=self.weights)),
            axis=1, arr=predictions)
    return maj_vote
```

<div style="text-align: right"><font size="4">9</font></div>

---
### Combining Three Classifiers

Using the Iris dataset with three base classifiers:

```python
clf1 = LogisticRegression(penalty='l2', C=0.001)

clf2 = DecisionTreeClassifier(max_depth=1,
                              criterion='entropy')

clf3 = KNeighborsClassifier(n_neighbors=1,
                            p=2, metric='minkowski')

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])
```

<div style="text-align: right"><font size="4">10</font></div>

---
### Evaluating Individual Classifiers

```python
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train, y=y_train,
                             cv=10, scoring='roc_auc')
    print(f'ROC AUC: {scores.mean():.2f} '
          f'(+/- {scores.std():.2f}) [{label}]')
```

| Classifier | ROC AUC |
|:-----------|:-------:|
| Logistic Regression | 0.87 (+/- 0.17) |
| Decision Tree | 0.89 (+/- 0.16) |
| KNN | 0.88 (+/- 0.15) |

<div style="text-align: right"><font size="4">11</font></div>

---
### Majority Voting Performance

```python
mv_clf = MajorityVoteClassifier(
    classifiers=[pipe1, clf2, pipe3])

scores = cross_val_score(estimator=mv_clf,
                         X=X_train, y=y_train,
                         cv=10, scoring='roc_auc')
```

| Classifier | ROC AUC |
|:-----------|:-------:|
| Logistic Regression | 0.87 |
| Decision Tree | 0.89 |
| KNN | 0.88 |
| **Majority Voting** | **0.94** |

<div style="text-align: right"><font size="4">12</font></div>

---
### Tuning the Ensemble with GridSearchCV

```python
params = {
    'decisiontreeclassifier__max_depth': [1, 2],
    'pipeline-1__clf__C': [0.001, 0.1, 100.0]
}

grid = GridSearchCV(estimator=mv_clf,
                    param_grid=params,
                    cv=10, scoring='roc_auc')
grid.fit(X_train, y_train)

print(f'Best parameters: {grid.best_params_}')
print(f'ROC AUC: {grid.best_score_:.2f}')
```

<div style="text-align: right"><font size="4">13</font></div>

---
### Bagging: Bootstrap Aggregating

**Core Idea:** Train multiple models on different bootstrap samples

**Bootstrap Sample:** Random sample with replacement from training data

**Aggregation:**
- Regression: Average predictions
- Classification: Majority vote

<div style="text-align: right"><font size="4">14</font></div>

---
### Why Bagging Works

1. Each bootstrap sample contains ~63.2% of original data
2. Different samples create diverse models
3. Models make different errors on different regions
4. Averaging reduces variance without increasing bias

**Best for:** High-variance, low-bias models (e.g., unpruned decision trees)

<div style="text-align: right"><font size="4">15</font></div>

---
### Bagging with Wine Dataset

```python
from sklearn.ensemble import BaggingClassifier

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=None,
                              random_state=1)

bag = BaggingClassifier(estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        n_jobs=1,
                        random_state=1)
```

<div style="text-align: right"><font size="4">16</font></div>

---
### Bagging Results

```python
tree = tree.fit(X_train, y_train)
tree_train = accuracy_score(y_train, tree.predict(X_train))
tree_test = accuracy_score(y_test, tree.predict(X_test))

bag = bag.fit(X_train, y_train)
bag_train = accuracy_score(y_train, bag.predict(X_train))
bag_test = accuracy_score(y_test, bag.predict(X_test))
```

| Model | Train Accuracy | Test Accuracy |
|:------|:--------------:|:-------------:|
| Decision Tree | 1.000 | 0.833 |
| **Bagging (500 trees)** | **1.000** | **0.917** |

<div style="text-align: right"><font size="4">17</font></div>

---
### Adaptive Boosting (AdaBoost)

**Key Difference from Bagging:**
- Bagging: Models trained independently (parallel)
- Boosting: Models trained sequentially, each focusing on previous errors

**AdaBoost Process:**
1. Train weak learner on weighted data
2. Increase weights of misclassified samples
3. Train next learner on reweighted data
4. Combine learners with weighted vote

<div style="text-align: right"><font size="4">18</font></div>

---
### AdaBoost Weight Updates

Given predictions and true labels:

```python
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1, -1])
yhat = np.array([1, 1, 1, -1, -1, -1, -1, -1, -1, -1])
correct = (y == yhat)
weights = np.full(10, 0.1)  # Initial uniform weights
```

**Weighted error rate:**
```python
epsilon = np.mean(~correct)  # 0.3
```

<div style="text-align: right"><font size="4">19</font></div>

---
### AdaBoost: Classifier Weight

Classifier weight based on its performance:

$$\alpha_j = \frac{1}{2} \ln\left(\frac{1-\epsilon}{\epsilon}\right)$$

```python
alpha_j = 0.5 * np.log((1-epsilon) / epsilon)
# alpha_j = 0.424
```

**Interpretation:** Better classifiers get higher weights

<div style="text-align: right"><font size="4">20</font></div>

---
### AdaBoost: Sample Weight Updates

```python
# Update weights
update_if_correct = 0.1 * np.exp(-alpha_j * 1 * 1)
# 0.065

update_if_wrong = 0.1 * np.exp(-alpha_j * 1 * -1)
# 0.153

# Normalize weights to sum to 1
weights = np.where(correct, update_if_correct,
                   update_if_wrong)
normalized_weights = weights / np.sum(weights)
```

**Effect:** Misclassified samples get higher weights

<div style="text-align: right"><font size="4">21</font></div>

---
### AdaBoost with Scikit-Learn

```python
from sklearn.ensemble import AdaBoostClassifier

tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=1,  # Weak learner!
                              random_state=1)

ada = AdaBoostClassifier(estimator=tree,
                         n_estimators=500,
                         learning_rate=0.1,
                         random_state=1)
```

**Note:** AdaBoost uses shallow trees (stumps) as weak learners

<div style="text-align: right"><font size="4">22</font></div>

---
### AdaBoost Results

```python
tree = tree.fit(X_train, y_train)
ada = ada.fit(X_train, y_train)
```

| Model | Train Accuracy | Test Accuracy |
|:------|:--------------:|:-------------:|
| Decision Stump | 0.845 | 0.792 |
| **AdaBoost (500 stumps)** | **1.000** | **0.917** |

**Key:** Many weak learners combine into a strong learner

<div style="text-align: right"><font size="4">23</font></div>

---
### Gradient Boosting

**Similar to AdaBoost but:**
- Fits new models to residual errors (gradients)
- Uses gradient descent to minimize loss function
- More flexible loss functions possible

**Key Hyperparameters:**
- `n_estimators`: Number of boosting stages
- `learning_rate`: Shrinkage parameter
- `max_depth`: Depth of individual trees

<div style="text-align: right"><font size="4">24</font></div>

---
### XGBoost: Extreme Gradient Boosting

```python
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=4,
    random_state=1,
    use_label_encoder=False
)

gbm = model.fit(X_train, y_train)
```

**Advantages:** Speed, regularization, handling missing values

<div style="text-align: right"><font size="4">25</font></div>

---
### XGBoost Results

```python
y_train_pred = gbm.predict(X_train)
y_test_pred = gbm.predict(X_test)

gbm_train = accuracy_score(y_train, y_train_pred)
gbm_test = accuracy_score(y_test, y_test_pred)
```

| Model | Train Accuracy | Test Accuracy |
|:------|:--------------:|:-------------:|
| Decision Tree | 1.000 | 0.833 |
| Bagging | 1.000 | 0.917 |
| AdaBoost | 1.000 | 0.917 |
| **XGBoost** | **1.000** | **0.958** |

<div style="text-align: right"><font size="4">26</font></div>

---
### Comparison: Bagging vs Boosting

| Aspect | Bagging | Boosting |
|:-------|:--------|:---------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Base learners | Complex (deep trees) | Simple (stumps) |
| Overfitting | Resistant | Can overfit |
| Data sensitivity | Robust to noise | Sensitive to outliers |

<div style="text-align: right"><font size="4">27</font></div>

---
### When to Use Each Method

**Bagging / Random Forests:**
- High variance models
- Noisy data
- Need interpretability

**AdaBoost / Gradient Boosting:**
- High bias models
- Clean data
- Maximum predictive performance

<div style="text-align: right"><font size="4">28</font></div>

---
### Summary

- **Ensembles** combine multiple models to improve performance
- **Majority voting** combines classifiers through votes or probabilities
- **Bagging** reduces variance by training on bootstrap samples
- **AdaBoost** sequentially focuses on misclassified examples
- **Gradient boosting** minimizes loss by fitting to residuals
- **XGBoost** provides optimized gradient boosting with regularization

<div style="text-align: right"><font size="4">29</font></div>

---
### References

- Raschka, S. & Mirjalili, V. (2022). *Machine Learning with PyTorch and Scikit-Learn*. Packt Publishing. Chapter 7.
- Code: https://github.com/rasbt/machine-learning-book/tree/main/ch07
- Breiman, L. (1996). Bagging Predictors. *Machine Learning*.
- Freund, Y. & Schapire, R. (1997). A Decision-Theoretic Generalization of On-Line Learning. *JCSS*.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *KDD*.

<div style="text-align: right"><font size="4">30</font></div>
