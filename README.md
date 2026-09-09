# Irredundant-k-fold-cross-validation
Irredundant k-Fold Cross-Validation (IkF) is a scikit-learn-compatible CV splitter designed to eliminate training-set reuse across folds.  In standard k-f CV, each observation is used for training in k-1 different iterations. In IkF, each observation is used exactly once for training and once for testing over the complete validation procedure.
