# Ensemble Methods for Tabular Data
### CAP6606: Machine Learning for ISR
#### Dr. Brian Jalaian

<div style="text-align: right"><font size="4">1</font></div>

---
### Learning Objectives

- Explain mathematical foundations of combining weak learners
- Implement voting ensembles and understand their benefits
- Apply bagging techniques and bootstrap sampling
- Master Random Forests and key hyperparameter tuning
- Interpret models using feature importance methods
- Assess prediction confidence through tree disagreement
- Understand extrapolation limitations of tree-based methods

<div style="text-align: right"><font size="4">2</font></div>

---
### Why Ensembles Work

**Core Principle:** Error Diversification

> "If each model is better than random guessing and makes different mistakes, combining them yields higher accuracy."

| Metric | Single Classifier | Ensemble (11 classifiers) |
|:-------|:-----------------:|:-------------------------:|
| Error Rate | 25% | 3.4% |
| Improvement | - | **86% reduction** |

**Key requirement:** Uncorrelated errors among base learners

<div style="text-align: right"><font size="4">3</font></div>

---
### The Math Behind Ensemble Success

For $n$ independent classifiers with error probability $\epsilon$:

$$P(\text{majority wrong}) = \sum_{k > n/2}^{n} \binom{n}{k} \epsilon^k (1-\epsilon)^{n-k}$$

**Example:** 11 classifiers, each with 25% error:

```python
from scipy.stats import binom
n, epsilon = 11, 0.25
ensemble_error = 1 - binom.cdf(n//2, n, epsilon)
print(f"Ensemble error: {ensemble_error:.3f}")  # 0.034
```

<div style="text-align: right"><font size="4">4</font></div>

---
### Decision Trees as Building Blocks

**How Trees Work:**
- Recursively partition feature space through yes/no questions
- Minimize impurity at each split (Gini or entropy)
- Terminal leaves contain predictions

**The Instability Problem:**
- High variance: small data changes produce different trees
- This weakness becomes a **strength** for ensembles!

<div style="text-align: right"><font size="4">5</font></div>

---
### Bagging: Bootstrap Aggregating

**Process:**
1. Create $B$ bootstrap samples (random sampling with replacement)
2. Train one model on each bootstrap sample
3. Average predictions (regression) or vote (classification)

**Why It Works:**

> "Each tree has errors, but those errors are random and uncorrelated. The average of random errors is zero."

<div style="text-align: right"><font size="4">6</font></div>

---
### Bootstrap Sampling Illustrated

Original data: $[1, 2, 3, 4, 5]$

| Bootstrap Sample | Contents |
|:---------------:|:---------|
| Sample 1 | $[1, 1, 3, 4, 5]$ |
| Sample 2 | $[2, 2, 3, 4, 4]$ |
| Sample 3 | $[1, 3, 3, 5, 5]$ |

**Key insight:** ~63.2% of original data appears in each sample

The remaining ~36.8% becomes **Out-of-Bag (OOB)** data

<div style="text-align: right"><font size="4">7</font></div>

---
### Out-of-Bag Error: Free Validation

```python
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

bagging = BaggingRegressor(
    estimator=DecisionTreeRegressor(),
    n_estimators=50,
    oob_score=True,  # Enable OOB scoring
    random_state=42
)
bagging.fit(X_train, y_train)
print(f"OOB R²: {bagging.oob_score_:.4f}")
```

**Advantage:** No need for separate validation set!

<div style="text-align: right"><font size="4">8</font></div>

---
### Random Forests: Beyond Bagging

**Key Innovation:** Random feature selection at each split

Instead of considering all $p$ features:
- Classification: randomly select $\sqrt{p}$ features
- Regression: randomly select $p/3$ features

**Effect:** Forces trees to use different features, ensuring diversity

<div style="text-align: right"><font size="4">9</font></div>

---
### Random Forest Implementation

```python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=100,      # Number of trees
    max_features='sqrt',   # Features per split
    min_samples_leaf=5,    # Regularization
    oob_score=True,
    n_jobs=-1,             # Parallel processing
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB R²: {rf.oob_score_:.4f}")
```

<div style="text-align: right"><font size="4">10</font></div>

---
### Key Random Forest Properties

| Property | Implication |
|:---------|:------------|
| Cannot overfit by adding trees | Safe to use many trees |
| OOB provides free validation | No train/val split needed |
| 100-200 trees typically sufficient | Diminishing returns beyond |
| Parallelizable | Fast training |

<div style="text-align: right"><font size="4">11</font></div>

---
### Performance Comparison

| Model | R² Score |
|:------|:--------:|
| Single Decision Tree | 0.5997 |
| Bagging (50 trees) | 0.6469 |
| **Random Forest (100 trees)** | **0.7734** |
| Gradient Boosting | 0.8160 |

Dataset: California Housing (20,640 samples, 8 features)

<div style="text-align: right"><font size="4">12</font></div>

---
### Feature Importance: Impurity-Based

**Default sklearn method:**
- Measures how much each feature decreases impurity
- Summed across all splits where feature is used

```python
importances = rf.feature_importances_
for name, imp in zip(feature_names, importances):
    print(f"{name}: {imp:.4f}")
```

**Warning:** Biased toward high-cardinality features!

<div style="text-align: right"><font size="4">13</font></div>

---
### Permutation Importance: More Reliable

**Algorithm:**
1. Compute baseline model score
2. For each feature: shuffle values, recompute score
3. Importance = score drop after shuffling

> "If shuffling a feature doesn't hurt performance, that feature isn't important."

```python
from sklearn.inspection import permutation_importance

result = permutation_importance(rf, X_val, y_val,
                                 n_repeats=10)
```

<div style="text-align: right"><font size="4">14</font></div>

---
### Partial Dependence Plots

Show marginal effect of features on predictions:

```python
from sklearn.inspection import PartialDependenceDisplay

PartialDependenceDisplay.from_estimator(
    rf, X_train,
    features=['MedInc', 'AveRooms'],
    kind='both'  # Show individual + average
)
```

**Use case:** Understanding non-linear relationships

<div style="text-align: right"><font size="4">15</font></div>

---
### Prediction Confidence

**Idea:** Tree disagreement indicates uncertainty

```python
# Get predictions from each tree
tree_preds = np.array([
    tree.predict(X_new)
    for tree in rf.estimators_
])

# High std = low confidence
confidence = 1 / (1 + tree_preds.std(axis=0))
```

**Application:** Flag predictions needing human review

<div style="text-align: right"><font size="4">16</font></div>

---
### Extrapolation Limitation

**Critical weakness:** Trees cannot extrapolate!

- Predictions are averages of training leaf values
- Cannot output values outside training data range
- Problematic for:
  - Time series with trends
  - Extreme feature values

**Solution:** Consider gradient boosting or neural networks for extrapolation tasks

<div style="text-align: right"><font size="4">17</font></div>

---
### Gradient Boosting: Sequential Learning

**Algorithm:**
1. Start with simple prediction (mean)
2. Calculate residuals (errors)
3. Train small tree on residuals
4. Add scaled predictions to ensemble
5. Repeat with new residuals

$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$

Where $\eta$ is the learning rate

<div style="text-align: right"><font size="4">18</font></div>

---
### Gradient Boosting Implementation

```python
from sklearn.ensemble import GradientBoostingRegressor

gb = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,           # Shallow trees
    min_samples_leaf=5,
    validation_fraction=0.1,
    n_iter_no_change=10,   # Early stopping
    random_state=42
)
gb.fit(X_train, y_train)
```

<div style="text-align: right"><font size="4">19</font></div>

---
### Random Forest vs Gradient Boosting

| Aspect | Random Forest | Gradient Boosting |
|:-------|:--------------|:------------------|
| Training | Parallel | Sequential |
| Tree depth | Deep | Shallow (3-6 levels) |
| Tuning | Easy, robust | Requires care |
| Overfitting | Resistant | Needs early stopping |
| Performance | Good baseline | Often slightly better |

<div style="text-align: right"><font size="4">20</font></div>

---
### When to Use Tree Ensembles

**Best for:**
- Tabular/structured data
- Mixed feature types (numeric + categorical)
- Missing values and outliers
- Interpretability requirements
- Fast iteration and prototyping

**Consider alternatives for:**
- Images, text, audio (use neural networks)
- Extrapolation requirements
- Very high-dimensional sparse data

<div style="text-align: right"><font size="4">21</font></div>

---
### Practical Recommendations

1. **Start with Random Forest** - good baseline, minimal tuning
2. **Use OOB score** for quick validation
3. **Check permutation importance** for feature selection
4. **Monitor tree disagreement** for prediction confidence
5. **Try Gradient Boosting** (XGBoost, LightGBM) for final push
6. **Watch for extrapolation** issues in production

<div style="text-align: right"><font size="4">22</font></div>

---
### Key Takeaways

- **Ensembles work** because they average out uncorrelated errors
- **Bagging** creates diversity through bootstrap sampling
- **Random Forests** add feature randomization for more diversity
- **Interpretation tools** make tree ensembles transparent
- **Gradient Boosting** trades ease-of-use for potential gains
- **Know the limitations:** trees cannot extrapolate

<div style="text-align: right"><font size="4">23</font></div>

---
### References

- Breiman, L. (1996). Bagging Predictors. *Machine Learning*, 24(2), 123-140.
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics*.
- Fast.ai Practical Deep Learning Course
- XGBoost & LightGBM Documentation

<div style="text-align: right"><font size="4">24</font></div>
