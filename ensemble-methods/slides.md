# Combining Different Models for Ensemble Learning
### CAP6606: Machine Learning for ISR
#### Professor Brian Jalaian
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

- $\binom{n}{k}$: ways to choose which $k$ classifiers are wrong
- $\epsilon^k$: probability those $k$ are all wrong
- $(1-\epsilon)^{n-k}$: probability the rest are right
- Sum over all $k > n/2$ (majority wrong)

<div style="text-align: right"><font size="4">4</font></div>

---
### Ensemble Error vs Base Error

With $n = 11$ classifiers, the ensemble error drops dramatically:

| Base Error ($\epsilon$) | Ensemble Error |
|:----------:|:---------------------:|
| 0.25 | **0.034** |
| 0.35 | 0.133 |
| 0.45 | 0.417 |
| 0.50 | 0.500 |

**Key:** Ensembles only help when base error < 0.5

<div style="text-align: right"><font size="4">5</font></div>

---
### Combining Classifiers via Majority Vote

**Hard Voting — Majority Label:**

$$\hat{y} = \text{mode}\ \lbrace h_1(\mathbf{x}),\ h_2(\mathbf{x}),\ \ldots,\ h_n(\mathbf{x}) \rbrace$$

| Classifier | Vote |
|:----------:|:----:|
| Classifier 1 | Class A |
| Classifier 2 | Class A |
| Classifier 3 | Class B |
| **Ensemble** | **→ Class A** (2 votes) |

**Soft Voting:** average predicted *probabilities* instead of labels — uses more information

<div style="text-align: right"><font size="4">6</font></div>

---
### Soft Voting Example

Three classifiers with weights 0.2, 0.2, 0.6:

| Classifier | Weight | P(Class A) | P(Class B) |
|:----------:|:------:|:----------:|:----------:|
| Classifier 1 | 0.2 | 0.90 | 0.10 |
| Classifier 2 | 0.2 | 0.80 | 0.20 |
| Classifier 3 | **0.6** | 0.40 | **0.60** |
| **Weighted avg** | | **0.58** | **0.42** |
| **Prediction** | | **→ Class A** | |

Even though the highest-weighted classifier voted B, the ensemble picks A

<div style="text-align: right"><font size="4">7</font></div>

---
### Majority Voting — Formal Definition

**Hard vote (weighted mode):**
$$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{j=1}^{m} w_j \mathbf{1}\left[h_j(\mathbf{x}) = c\right]$$

**Soft vote (weighted posterior average):**
$$\hat{y} = \arg\max_{c \in \mathcal{C}} \sum_{j=1}^{m} w_j p_j(c \mid \mathbf{x})$$

$h_j$ — $j$-th classifier &nbsp;|&nbsp; $w_j \geq 0,\ \sum_j w_j = 1$ — classifier weights &nbsp;|&nbsp; $p_j(c \mid \mathbf{x})$ — posterior estimate

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Kuncheva (2004), Combining Pattern Classifiers, §4</div>
<div style="text-align: right"><font size="4">8</font></div>

---
### When Does Soft Voting Beat Hard?

**Result (Kittler, Hatef, Duin & Matas 1998):** if base classifiers output conditionally-independent, well-calibrated posteriors $p_j(c \mid \mathbf{x})$, the weighted-average soft vote converges to the Bayes-optimal classifier as $m \to \infty$.

| Assumption | Hard voting | Soft voting |
|:--|:--:|:--:|
| Calibrated posteriors required | not used | required for optimality |
| Confidence preserved | discarded | averaged |
| Robust to a single over-confident bad classifier | yes | no — one bad probability can swing the average |
| Works when base outputs only labels | yes | n/a |

**Practical rule:** calibrate base learners (e.g. Platt scaling, isotonic regression), then prefer soft voting. Fall back to hard voting when posteriors are unreliable.

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Kittler, Hatef, Duin & Matas (1998), On Combining Classifiers, IEEE TPAMI 20(3)</div>
<div style="text-align: right"><font size="4">9</font></div>

---
### Empirical Demonstration — Majority Voting

**Setup:** Iris dataset (3 classes, 4 features); three deliberately diverse base learners — logistic regression, decision tree, K-nearest neighbors — combined via soft voting. Evaluated by 10-fold cross-validation, ROC-AUC.

**Purpose of this empirical check:** test two predictions of the theory on the previous slides.

1. The ensemble should outperform every individual base learner (diversity → uncorrelated errors).
2. Soft voting should beat hard voting when base learners are calibrated.

<div style="background: rgba(255, 184, 108, 0.10); border-left: 4px solid #ffb86c; padding: 12px 18px; margin-top: 14px; font-size: 0.78em; text-align: left;">
<b style="color: #ffb86c;">🧪 Hands-on checkpoint —</b> Pause here and run this in <code>01_ensemble_methods.ipynb</code> (<em>Majority Voting</em> section: code, per-classifier metrics, grid-searched hyperparameters, learning curves), <b>or</b> keep watching the deck — we'll walk through the notebook together in a separate video.
</div>

<div style="text-align: right"><font size="4">10</font></div>

---
### Bagging — Algorithm

**Bootstrap Aggregating (Breiman 1996):** reduce variance by averaging predictors trained on resampled data.

<div style="font-family: monospace; font-size: 0.7em; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 16px; border-radius: 6px; line-height: 1.55;">
<b>Algorithm: Bagging</b><br>
<b>Input:</b> training set $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{n}$, base learner $\mathcal{L}$, ensemble size $M$<br>
<b>For</b> $m = 1, \ldots, M$:<br>
&nbsp;&nbsp;1. Draw bootstrap sample $\mathcal{D}_m$ of size $n$ from $\mathcal{D}$ (sample with replacement)<br>
&nbsp;&nbsp;2. Train base learner: $h_m \leftarrow \mathcal{L}(\mathcal{D}_m)$<br>
<b>Return:</b> $\hat{F}(\mathbf{x}) = \frac{1}{M} \sum_{m=1}^{M} h_m(\mathbf{x})$ (regression) &nbsp;or&nbsp; majority vote (classification)
</div>

Probability a sample is *not* in a bootstrap: $\left(1 - \tfrac{1}{n}\right)^n \to \tfrac{1}{e} \approx 0.368$ &nbsp;→&nbsp; each $h_m$ sees ~63.2% of $\mathcal{D}$ (the remaining ~36.8% is the *out-of-bag* set, free validation).

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Breiman (1996), Bagging Predictors, Machine Learning 24(2)</div>
<div style="text-align: right"><font size="4">11</font></div>

---
### Why Bagging Works — Bias–Variance View

For an ensemble of $M$ predictors with pairwise correlation $\rho$ and per-predictor variance $\sigma^2$:

$$\operatorname{Var}\left[\hat{F}(\mathbf{x})\right] = \rho \sigma^2 + \frac{1 - \rho}{M} \sigma^2$$

| Limit | Variance behavior |
|:--|:--|
| $\rho \to 0$ (uncorrelated) | $\operatorname{Var} \to \sigma^2 / M$ — unbounded reduction with $M$ |
| $\rho \to 1$ (identical predictors) | $\operatorname{Var} \to \sigma^2$ — no reduction at all |
| Bias | unchanged by averaging |

**Design implication:** the goal of bagging — and of Random Forests — is to *decorrelate* the base learners. Bootstrap sampling alone shrinks $\rho$ only slightly; the per-split random feature subset in Random Forests is what really drives $\rho$ down.

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Hastie, Tibshirani & Friedman (2009), Elements of Statistical Learning, §§8.7, 15.4.1</div>
<div style="text-align: right"><font size="4">12</font></div>

---
### Empirical Demonstration — Bagging on Wine

**Setup:** Wine dataset (178 samples, 13 features, 3 classes); base learner = unpruned decision tree (high variance, near-zero bias); ensemble of 500 trees.

**Purpose:** validate the prediction of the bias–variance equation — that bagging reduces variance without inflating bias. Compare a single overfit tree to the bagged ensemble; expect identical training accuracy and a large test-accuracy gap.

<div style="background: rgba(255, 184, 108, 0.10); border-left: 4px solid #ffb86c; padding: 12px 18px; margin-top: 14px; font-size: 0.78em; text-align: left;">
<b style="color: #ffb86c;">🧪 Hands-on checkpoint —</b> Pause here and run this in <code>01_ensemble_methods.ipynb</code> (<em>Bagging</em> section: implementation, learning curves, out-of-bag error estimate, decision-boundary plot), <b>or</b> keep watching the deck — we'll walk through the notebook together in a separate video.
</div>

<div style="text-align: right"><font size="4">13</font></div>

---
### AdaBoost — Algorithm

**Adaptive Boosting (Freund & Schapire 1997):** sequentially train weak learners on a distribution that emphasizes the mistakes of previous rounds.

<div style="font-family: monospace; font-size: 0.68em; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 16px; border-radius: 6px; line-height: 1.55;">
<b>Algorithm: AdaBoost (binary, $y_i \in \{-1, +1\}$)</b><br>
<b>Init:</b> $w_i^{(1)} = 1/n$ for $i = 1, \ldots, n$<br>
<b>For</b> $m = 1, \ldots, M$:<br>
&nbsp;&nbsp;1. Fit weak learner $h_m(\mathbf{x}) \in \{-1, +1\}$ on $\mathcal{D}$ with weights $\mathbf{w}^{(m)}$<br>
&nbsp;&nbsp;2. Weighted error: $\epsilon_m = \sum_{i=1}^{n} w_i^{(m)} \mathbf{1}[h_m(\mathbf{x}_i) \neq y_i]$<br>
&nbsp;&nbsp;3. Learner weight: $\alpha_m = \tfrac{1}{2} \ln\left(\dfrac{1 - \epsilon_m}{\epsilon_m}\right)$<br>
&nbsp;&nbsp;4. Update sample weights: $w_i^{(m+1)} \propto w_i^{(m)} \exp\left(-\alpha_m y_i h_m(\mathbf{x}_i)\right)$ &nbsp; (then renormalize)<br>
<b>Return:</b> $H(\mathbf{x}) = \operatorname{sign}\left(\sum_{m=1}^{M} \alpha_m h_m(\mathbf{x})\right)$
</div>

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Freund & Schapire (1997), J. Comput. Syst. Sci. 55(1)</div>
<div style="text-align: right"><font size="4">14</font></div>

---
### Where Do These Update Rules Come From?

**AdaBoost = forward-stagewise additive modeling under exponential loss** (Friedman, Hastie & Tibshirani 2000). With $\mathcal{L}\_{\exp}(y, F) = e^{-yF}$, at round $m$ solve:

$$(\alpha\_m, h\_m) = \arg\min\_{\alpha, h} \sum\_{i=1}^{n} \exp\left(-y\_i \big[F\_{m-1}(\mathbf{x}\_i) + \alpha h(\mathbf{x}\_i)\big]\right)$$

Letting $w\_i^{(m)} \propto \exp(-y\_i F\_{m-1}(\mathbf{x}\_i))$ — the *exponential-loss gradients at the current ensemble* — the inner minimization yields the closed-form $\alpha\_m = \tfrac{1}{2}\ln\big((1 - \epsilon\_m) / \epsilon\_m\big)$ and the multiplicative weight update from the previous slide.

**Consequence:** AdaBoost's label-noise sensitivity traces directly to the exponential loss — outliers attract exponentially growing weight.

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Friedman, Hastie & Tibshirani (2000), Additive Logistic Regression: A Statistical View of Boosting, Annals of Statistics 28(2)</div>
<div style="text-align: right"><font size="4">15</font></div>

---
### AdaBoost — Boundary Evolution

<div style="display: flex; justify-content: space-around; align-items: flex-start; margin-top: 10px;">
<div style="text-align: center; width: 30%;">
<svg viewBox="0 0 200 200" style="width: 100%; height: auto; background: #fafafa; border: 1px solid #ccc;">
<line x1="0" y1="100" x2="200" y2="100" stroke="#444" stroke-width="2"/>
<circle cx="40" cy="60" r="5" fill="#1f77b4"/>
<circle cx="80" cy="50" r="5" fill="#1f77b4"/>
<circle cx="120" cy="70" r="5" fill="#1f77b4"/>
<circle cx="160" cy="55" r="5" fill="#1f77b4"/>
<circle cx="50" cy="140" r="5" fill="#d62728"/>
<circle cx="100" cy="155" r="5" fill="#d62728"/>
<circle cx="150" cy="135" r="5" fill="#d62728"/>
<circle cx="65" cy="120" r="5" fill="#d62728"/>
<circle cx="135" cy="85" r="5" fill="#d62728"/>
<circle cx="90" cy="115" r="5" fill="#1f77b4"/>
</svg>
<b>Round 1</b><br><font size="3">uniform weights<br>single stump</font>
</div>
<div style="text-align: center; width: 30%;">
<svg viewBox="0 0 200 200" style="width: 100%; height: auto; background: #fafafa; border: 1px solid #ccc;">
<line x1="0" y1="100" x2="200" y2="100" stroke="#aaa" stroke-width="2" stroke-dasharray="4 2"/>
<line x1="118" y1="0" x2="118" y2="200" stroke="#444" stroke-width="2"/>
<circle cx="40" cy="60" r="4" fill="#1f77b4"/>
<circle cx="80" cy="50" r="4" fill="#1f77b4"/>
<circle cx="120" cy="70" r="4" fill="#1f77b4"/>
<circle cx="160" cy="55" r="4" fill="#1f77b4"/>
<circle cx="50" cy="140" r="4" fill="#d62728"/>
<circle cx="100" cy="155" r="4" fill="#d62728"/>
<circle cx="150" cy="135" r="4" fill="#d62728"/>
<circle cx="65" cy="120" r="4" fill="#d62728"/>
<circle cx="135" cy="85" r="10" fill="#d62728"/>
<circle cx="90" cy="115" r="10" fill="#1f77b4"/>
</svg>
<b>Round 5</b><br><font size="3">weights concentrate<br>on hard points</font>
</div>
<div style="text-align: center; width: 30%;">
<svg viewBox="0 0 200 200" style="width: 100%; height: auto; background: #fafafa; border: 1px solid #ccc;">
<path d="M 0 80 Q 60 90 120 75 T 200 110" stroke="#444" stroke-width="2" fill="none"/>
<path d="M 110 0 Q 115 60 130 100 T 145 200" stroke="#888" stroke-width="1.5" fill="none" stroke-dasharray="3 2"/>
<path d="M 0 135 Q 50 130 90 135 T 200 145" stroke="#aaa" stroke-width="1.2" fill="none" stroke-dasharray="3 2"/>
<circle cx="40" cy="60" r="3" fill="#1f77b4"/>
<circle cx="80" cy="50" r="3" fill="#1f77b4"/>
<circle cx="120" cy="70" r="3" fill="#1f77b4"/>
<circle cx="160" cy="55" r="3" fill="#1f77b4"/>
<circle cx="50" cy="140" r="3" fill="#d62728"/>
<circle cx="100" cy="155" r="3" fill="#d62728"/>
<circle cx="150" cy="135" r="3" fill="#d62728"/>
<circle cx="65" cy="120" r="3" fill="#d62728"/>
<circle cx="135" cy="85" r="4" fill="#d62728"/>
<circle cx="90" cy="115" r="4" fill="#1f77b4"/>
</svg>
<b>Round 50</b><br><font size="3">composite $\alpha$-weighted<br>boundary fits all</font>
</div>
</div>

Marker size $\propto$ sample weight $w_i^{(m)}$. The accumulated classifier $H_m = \operatorname{sign}\left(\sum_{j \le m} \alpha_j h_j\right)$ becomes a piecewise function far more expressive than any single stump.

<div style="text-align: right"><font size="4">16</font></div>

---
### Empirical Demonstration — AdaBoost on Wine

**Setup:** same Wine dataset; base learner = decision stump (depth = 1) — intentionally a weak learner with high bias and low variance; ensemble of 500 stumps; learning rate 0.1.

**Purpose:** validate the central boosting prediction — sequential reweighting under exponential loss should *reduce bias*, the opposite axis from bagging's variance reduction. Expect a single stump to underfit, while the boosted ensemble reaches near-perfect training fit.

<div style="background: rgba(255, 184, 108, 0.10); border-left: 4px solid #ffb86c; padding: 12px 18px; margin-top: 14px; font-size: 0.78em; text-align: left;">
<b style="color: #ffb86c;">🧪 Hands-on checkpoint —</b> Pause here and run this in <code>01_ensemble_methods.ipynb</code> (<em>AdaBoost</em> section: implementation, learning-rate sweep, train-vs-test divergence, comparison to a single stump), <b>or</b> keep watching the deck — we'll walk through the notebook together in a separate video.
</div>

<div style="text-align: right"><font size="4">17</font></div>

---
### Boosting as Functional Gradient Descent

**Gradient boosting (Friedman 2001):** generalize AdaBoost to *any* differentiable loss $\ell(y, F)$ by viewing boosting as gradient descent in function space. At stage $m$, fit $h\_m$ to the pseudo-residual (negative gradient at $F\_{m-1}$), then take a learning-rate-scaled step:

$$r\_{im} = - \frac{\partial \ell(y\_i, F\_{m-1}(\mathbf{x}\_i))}{\partial F\_{m-1}(\mathbf{x}\_i)}, \qquad F\_m(\mathbf{x}) = F\_{m-1}(\mathbf{x}) + \nu \gamma\_m h\_m(\mathbf{x})$$

**Special case — squared loss:** $\ell = \tfrac{1}{2}(y - F)^2 \implies -\partial_F \ell = y - F$ (the *residual*). So "fitting the residuals" is exactly gradient boosting under $L_2$ loss; other losses give different pseudo-residuals — log-loss for classification, Huber for robust regression.

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Friedman (2001), Greedy Function Approximation: A Gradient Boosting Machine, Annals of Statistics 29(5)</div>
<div style="text-align: right"><font size="4">18</font></div>

---
### Gradient Boosting — Algorithm

<div style="font-family: monospace; font-size: 0.68em; background: #0d1117; color: #c9d1d9; border: 1px solid #30363d; padding: 16px; border-radius: 6px; line-height: 1.55;">
<b>Algorithm: Gradient Boosting</b><br>
<b>Input:</b> training set $\mathcal{D}$, loss $\ell$, base-learner family $\mathcal{H}$, rounds $M$, learning rate $\nu$<br>
<b>Init:</b> $F_0(\mathbf{x}) = \arg\min_{c}  \sum_{i=1}^{n} \ell(y_i, c)$ &nbsp;(optimal constant)<br>
<b>For</b> $m = 1, \ldots, M$:<br>
&nbsp;&nbsp;1. Pseudo-residuals: $r_{im} = -\dfrac{\partial \ell(y_i, F_{m-1}(\mathbf{x}_i))}{\partial F_{m-1}(\mathbf{x}_i)}$<br>
&nbsp;&nbsp;2. Fit base learner: $h_m = \arg\min_{h \in \mathcal{H}} \sum_{i=1}^{n} \left(r_{im} - h(\mathbf{x}_i)\right)^2$<br>
&nbsp;&nbsp;3. Line search: $\gamma_m = \arg\min_{\gamma}  \sum_{i=1}^{n} \ell\left(y_i,  F_{m-1}(\mathbf{x}_i) + \gamma  h_m(\mathbf{x}_i)\right)$<br>
&nbsp;&nbsp;4. Update: $F_m(\mathbf{x}) = F_{m-1}(\mathbf{x}) + \nu \gamma_m h_m(\mathbf{x})$<br>
<b>Return:</b> $F_M(\mathbf{x})$
</div>

| Hyperparameter | Effect |
|:--|:--|
| $M$ (rounds) | additive capacity; large $M$ with small $\nu$ is the canonical sweet spot |
| $\nu$ (shrinkage) | smaller → better generalization, slower convergence |
| base-learner depth | typically 3–8 for trees; deeper captures interactions, raises variance |

<div style="text-align: right"><font size="4">19</font></div>

---
### XGBoost — Regularized Objective

XGBoost (Chen & Guestrin 2016) adds **explicit regularization** and a **second-order Taylor expansion** of the loss for faster, more accurate splits.

**Objective at round $t$:**

$$\mathcal{L}^{(t)} = \sum\_{i=1}^{n} \ell\left(y\_i, \hat{y}\_i^{(t-1)} + f\_t(\mathbf{x}\_i)\right) + \Omega(f\_t), \qquad \Omega(f) = \gamma T + \tfrac{1}{2} \lambda \|\mathbf{w}\|^2$$

$T$ — number of leaves &nbsp;|&nbsp; $\mathbf{w}$ — leaf weights &nbsp;|&nbsp; $\gamma$ — leaf penalty &nbsp;|&nbsp; $\lambda$ — $L\_2$ leaf-weight penalty

**Second-order Taylor expansion around $\hat{y}^{(t-1)}$:**

$$\mathcal{L}^{(t)} \approx \sum\_{i=1}^{n} \left[g\_i f\_t(\mathbf{x}\_i) + \tfrac{1}{2} h\_i f\_t(\mathbf{x}\_i)^2\right] + \Omega(f\_t), \qquad g\_i = \partial\_F \ell, \quad h\_i = \partial^2\_F \ell$$

**Closed-form optimal weight for leaf $j$ with sample set $I\_j$:**

$$w\_j^{*} = - \dfrac{\sum\_{i \in I\_j} g\_i}{\sum\_{i \in I\_j} h\_i + \lambda}$$

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Chen & Guestrin (2016), XGBoost: A Scalable Tree Boosting System, KDD '16</div>
<div style="text-align: right"><font size="4">20</font></div>

---
### XGBoost — Split Gain

Substituting the optimal leaf weights $w\_j^{*}$ back into the objective gives the **structure score** for a tree with leaf partition $\{I\_1, \ldots, I\_T\}$:

$$\tilde{\mathcal{L}}^{(t)} = -\tfrac{1}{2} \sum\_{j=1}^{T} \dfrac{\left(\sum\_{i \in I\_j} g\_i\right)^2}{\sum\_{i \in I\_j} h\_i + \lambda} + \gamma T$$

**Gain of splitting one leaf $I$ into $I\_L$ and $I\_R$:**

$$\text{Gain} = \tfrac{1}{2}\left[\dfrac{(\sum\_{i \in I\_L} g\_i)^2}{\sum\_{i \in I\_L} h\_i + \lambda} + \dfrac{(\sum\_{i \in I\_R} g\_i)^2}{\sum\_{i \in I\_R} h\_i + \lambda} - \dfrac{(\sum\_{i \in I} g\_i)^2}{\sum\_{i \in I} h\_i + \lambda}\right] - \gamma$$

A split is accepted only if $\text{Gain} > 0$ — the loss reduction must exceed the leaf-creation penalty $\gamma$. **This is XGBoost's built-in pre-pruning.**

The engineering layer (parallel histograms, sparsity-aware splits, cache-friendly access) sits on top of this mathematical core.

<div style="position: absolute; bottom: 10px; left: 20px; font-size: 0.45em; color: #888;">Chen & Guestrin (2016), §2.2 — "Learning the Tree Structure"</div>
<div style="text-align: right"><font size="4">21</font></div>

---
### Empirical Bake-Off — Bagging vs Boosting

**Setup:** Wine dataset; identical train/test split for all methods.
- Bagging — 500 unpruned trees
- AdaBoost — 500 decision stumps
- XGBoost — default hyperparameters

**Purpose:** illustrate that the bagging-vs-boosting choice is not academic — the same data combined with different inductive biases (variance reduction vs bias reduction) produces measurably different generalization.

<div style="background: rgba(255, 184, 108, 0.10); border-left: 4px solid #ffb86c; padding: 12px 18px; margin-top: 14px; font-size: 0.78em; text-align: left;">
<b style="color: #ffb86c;">🧪 Hands-on checkpoint —</b> Pause here and run this in <code>01_ensemble_methods.ipynb</code> (<em>Comparison</em> section: full bake-off table, learning curves, confusion matrices, significance check). The applied notebook <code>01b_ensemble_fraud_detection.ipynb</code> extends the comparison to credit-card fraud, where XGBoost is the production answer. <b>Or</b> keep watching the deck — both notebooks get their own walkthrough videos.
</div>

<div style="text-align: right"><font size="4">22</font></div>

---
### Bagging vs Boosting — Synthesis

| Axis | Bagging | Boosting |
|:--|:--|:--|
| Training | Parallel | Sequential |
| Target | Variance reduction | Bias reduction |
| Base learner preference | Low bias, high variance (deep trees) | High bias, low variance (stumps / shallow trees) |
| Sample weighting | Uniform (bootstrap only) | Adaptive (loss-driven) |
| Overfitting risk | Low (averaging) | Higher (can chase noisy labels) |
| Noise robustness | High | AdaBoost: low; GB with Huber loss: moderate |
| Parallelization | Trivial across trees | Per-round only (within histograms) |

**Mental model:** bagging is *Monte Carlo over models*; boosting is *gradient descent in function space*.

<div style="text-align: right"><font size="4">23</font></div>

---
### When to Use Which

**Bagging / Random Forests:**
- High-variance base learners (deep trees)
- Noisy labels or noisy features
- Feature-importance interpretability matters
- Small to moderate data; minimal hyperparameter tuning

**Gradient Boosting (XGBoost / LightGBM / CatBoost):**
- Maximum tabular-data accuracy
- Clean labels, time to tune hyperparameters
- Production deployment with calibration and monitoring

**On unstructured data** — images, audio, language — deep nets generally outperform both. But on tabular data, gradient-boosted trees remain the empirical default. *That is where we are now; deep learning is where we are headed.*

<div style="text-align: right"><font size="4">24</font></div>

---
### Summary

- Ensembles outperform single models by combining classifiers whose errors are weakly correlated.
- **Majority voting** is the simplest aggregator; *soft voting* is Bayes-optimal under independence and calibration (Kittler et al. 1998).
- **Bagging** reduces variance: $\operatorname{Var}_{\text{ensemble}} = \rho\sigma^2 + (1-\rho)\sigma^2/M$. Decorrelation drives the gain.
- **AdaBoost** is forward-stagewise minimization of exponential loss — $\alpha$ and weight updates fall out of the derivation (Friedman, Hastie & Tibshirani 2000).
- **Gradient Boosting** generalizes this to any differentiable loss via functional gradient descent (Friedman 2001).
- **XGBoost** is gradient boosting with explicit regularization, a second-order Taylor expansion for closed-form leaf weights, and a pre-pruning split criterion (Chen & Guestrin 2016).

<div style="text-align: right"><font size="4">25</font></div>

---
### References

<div style="font-size: 0.78em; text-align: left; max-width: 88%; margin: 0 auto;">

**Primary textbook:** Raschka & Mirjalili (2022). *Machine Learning with PyTorch and Scikit-Learn*, Ch. 7. Packt.

**Foundational papers:**
- Breiman (1996). Bagging Predictors. *Machine Learning* 24(2), 123–140.
- Freund & Schapire (1997). A Decision-Theoretic Generalization of On-Line Learning. *J. Comput. Syst. Sci.* 55(1), 119–139.
- Kittler, Hatef, Duin & Matas (1998). On Combining Classifiers. *IEEE TPAMI* 20(3), 226–239.
- Friedman, Hastie & Tibshirani (2000). Additive Logistic Regression. *Annals of Statistics* 28(2), 337–407.
- Friedman (2001). Greedy Function Approximation: A Gradient Boosting Machine. *Annals of Statistics* 29(5), 1189–1232.
- Chen & Guestrin (2016). XGBoost: A Scalable Tree Boosting System. *KDD '16*.

**Reference text:** Hastie, Tibshirani & Friedman (2009). *The Elements of Statistical Learning*, §§8.7, 10.1–10.5, 15.2–15.4. Springer.

**Code companion:** github.com/rasbt/machine-learning-book/tree/main/ch07

</div>

<div style="text-align: right"><font size="4">26</font></div>
