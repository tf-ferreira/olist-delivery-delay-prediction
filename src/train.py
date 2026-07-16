"""Temporal split and model pipelines for the delay classifier.

Three rules from the project's methodology are enforced here:

* **Temporal split, never random.** In production the model predicts the
  future from the past; a random split mixes future into training and
  overstates every metric.
* **Feature whitelist.** Only purchase-time columns (plus the hermetic
  seller-history features) may enter the model. The list below is the
  executable form of the leakage audit from notebook 01.
* **Train-only statistics.** Imputation (and scaling, for the linear model)
  lives INSIDE the sklearn pipeline, so medians/means are learned from the
  training fold alone. This closes the "no imputation in the master table"
  decision from the data prep stage.
"""

from __future__ import annotations

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "is_late"

# Purchase-time whitelist: the executable form of the leakage audit.
NUMERIC_FEATURES = [
    "promised_window_days",
    "purchase_month",
    "distance_km",
    "distance_missing",
    "n_items",
    "n_sellers",
    "total_price",
    "total_freight",
    "freight_price_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "weight_missing",
    "payment_total",
    "payment_installments_max",
    "payment_missing",
    "seller_late_rate_hist",
    "seller_posting_days_hist",
    "seller_no_history",
]
CATEGORICAL_FEATURES = ["customer_state", "main_category", "payment_type_primary"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def temporal_split(
    df: pd.DataFrame, cut: str, test_end: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by purchase timestamp: train strictly before ``cut``.

    ``test_end`` truncates the survivorship-biased tail measured in
    notebook 01 (delivery coverage collapses after Aug/2018); orders past it
    belong to neither side.

    Args:
        df: Order-level table with ``order_purchase_timestamp``.
        cut: First day of the test period (ISO date).
        test_end: First day EXCLUDED from the test period (ISO date).

    Returns:
        ``(train, test)`` DataFrames.
    """
    ts = df["order_purchase_timestamp"]
    train = df[ts < pd.Timestamp(cut)]
    test = df[(ts >= pd.Timestamp(cut)) & (ts < pd.Timestamp(test_end))]
    return train, test


def build_preprocessor(scale: bool) -> ColumnTransformer:
    """Preprocessing with train-only statistics.

    Args:
        scale: Standardize numerics (needed by the linear model; harmless
            but pointless for trees, so it is off for them).

    Returns:
        ColumnTransformer imputing numerics by median and one-hot encoding
        categoricals (unknown categories in validation/test are encoded as
        all-zeros instead of crashing).
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_model_grid(scale_pos_weight: float) -> dict[str, list[Pipeline]]:
    """Small, manual hyperparameter grid per model family.

    Class imbalance is handled by weights (never synthetic resampling): the
    real class distribution is preserved and the cost decision belongs to
    the business via the decision threshold, chosen later on validation.

    Args:
        scale_pos_weight: Ratio of negatives to positives in the TRAINING
            fold, passed to LightGBM (the tree analogue of class_weight).

    Returns:
        Mapping of family name to candidate pipelines (grids of 3-6 fits).
    """
    logreg = [
        Pipeline(
            [
                ("prep", build_preprocessor(scale=True)),
                (
                    "model",
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        for c in (0.1, 1.0, 10.0)
    ]
    forest = [
        Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        max_depth=depth,
                        min_samples_leaf=leaf,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        for depth, leaf in ((8, 20), (12, 20), (16, 5), (None, 20))
    ]
    lgbm = [
        Pipeline(
            [
                ("prep", build_preprocessor(scale=False)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=n,
                        learning_rate=0.05,
                        num_leaves=leaves,
                        min_child_samples=20,
                        scale_pos_weight=scale_pos_weight,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        verbosity=-1,
                    ),
                ),
            ]
        )
        for n, leaves in ((300, 15), (300, 31), (300, 63), (600, 31))
    ]
    return {"Regressão Logística": logreg, "Random Forest": forest, "LightGBM": lgbm}


def describe_params(pipeline: Pipeline) -> str:
    """Short human-readable id of a candidate's hyperparameters."""
    model = pipeline.named_steps["model"]
    if isinstance(model, LogisticRegression):
        return f"C={model.C}"
    if isinstance(model, RandomForestClassifier):
        return f"depth={model.max_depth}, leaf={model.min_samples_leaf}"
    return f"n={model.n_estimators}, leaves={model.num_leaves}"


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """AUC-PR (primary selection metric, rare positive class) and AUC-ROC."""
    proba = pipeline.predict_proba(X)[:, 1]
    return {
        "auc_pr": average_precision_score(y, proba),
        "auc_roc": roc_auc_score(y, proba),
    }
