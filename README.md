# Irredundant k-Fold Cross-Validation (IkF)

**Irredundant k-Fold Cross-Validation (IkF)** is a scikit-learn-compatible cross-validation splitter designed to eliminate training-set reuse across folds.

In standard k-fold cross-validation, each observation is used for training in `k-1` different iterations. In IkF, each observation is used **exactly once for training and once for testing** over the complete validation procedure.

Given `k` outer folds, each fold is divided into `k-1` stratified subfolds. When one outer fold is used for testing, the corresponding training set is formed by selecting one unused subfold from each of the remaining outer folds.

Therefore,

\[
|Test_i| \approx \frac{n}{k},
\qquad
|Train_i| \approx \frac{n}{k},
\]

and the training sets are mutually disjoint.

## Simple example

For `n = 18` samples and `k = 3`:

- the dataset is first divided into 3 folds of 6 samples;
- each fold is divided into `k-1 = 2` subfolds of approximately 3 samples;
- when Fold 1 is used for testing, one subfold from Fold 2 and one from Fold 3 are used for training;
- the selected subfolds are not reused in later training iterations.

Across the three iterations, every sample is therefore used once for testing and once for training.

## Usage

The API follows `scikit-learn`'s `StratifiedKFold`:

```python
from ikf import IStratifiedKFold

cv = IStratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for train_idx, test_idx in cv.split(X, y, rand=True):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
