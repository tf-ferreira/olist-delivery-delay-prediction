"""Final evaluation: metrics, operating points and the business scenario.

Discipline encoded here:

* The champion was selected on VALIDATION (stage 4). The test set reports;
  it never selects. Each model's decision threshold is also chosen on
  validation (max F1) and merely APPLIED to the test.
* The business scenario is expressed in COUNTS (customers notified, delays
  caught, false alarms), never in assumed money: counts are indisputable
  and the cost decision belongs to the business via the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)

OPERATING_BETAS = {"conservador": 0.5, "equilibrado": 1.0, "agressivo": 2.0}


def fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
    """F-beta score, elementwise and safe where precision = recall = 0."""
    b2 = beta * beta
    denom = b2 * precision + recall
    return np.divide(
        (1 + b2) * precision * recall,
        denom,
        out=np.zeros_like(denom),
        where=denom > 0,
    )


def pick_operating_points(y_true: pd.Series, proba: np.ndarray) -> pd.DataFrame:
    """Choose named operating points on the precision-recall curve.

    Uses F-beta maxima instead of arbitrary cutoffs: beta weighs recall
    against precision (0.5 favors precision, 2 favors recall), so each
    named point has a principled definition.

    Args:
        y_true: Binary target of the fold used to CHOOSE (validation).
        proba: Predicted probabilities on the same fold.

    Returns:
        One row per named point with threshold, precision and recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    rows = []
    for name, beta in OPERATING_BETAS.items():
        scores = fbeta(precision[:-1], recall[:-1], beta)
        i = int(np.argmax(scores))
        thr = float(thresholds[i])
        rows.append(
            {
                "ponto": name,
                "beta": beta,
                "threshold": thr,
                "precisão_val": float(precision[i]),
                "recall_val": float(recall[i]),
                # Alert budget: the fraction of orders alerted at this point.
                # Probability thresholds do NOT transfer across refits (the
                # probability scale shifts); the alert budget does, and it is
                # how an operation actually plans capacity.
                "alert_rate_val": float((proba >= thr).mean()),
            }
        )
    return pd.DataFrame(rows).set_index("ponto")


def threshold_for_alert_rate(proba: np.ndarray, rate: float) -> float:
    """Probability cutoff that alerts the top ``rate`` fraction of orders."""
    return float(np.quantile(proba, 1 - rate))


def threshold_metrics(
    y_true: pd.Series, proba: np.ndarray, threshold: float
) -> dict[str, float]:
    """Precision/recall/F1 of the hard decision at a given threshold."""
    alert = proba >= threshold
    tp = int((alert & (y_true == 1)).sum())
    fp = int((alert & (y_true == 0)).sum())
    fn = int((~alert & (y_true == 1)).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def final_comparison(
    finalists: dict[str, object],
    alert_rates: dict[str, float],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """The single-pass test table.

    Each model's operating point is its own max-F1 ALERT BUDGET, chosen
    out-of-sample on validation with the core-fitted model, and merely
    applied here (as the corresponding quantile of the model's own test
    scores). The test reports; it does not tune. Alert budgets rather than
    probability thresholds because probability scales shift across refits,
    while "alert the top k% riskiest orders" transfers unchanged.

    Args:
        finalists: Family name -> pipeline fitted on the FULL train.
        alert_rates: Family name -> alert budget chosen on validation.
        X_test: Test features (single pass).
        y_test: Test target.

    Returns:
        One row per finalist: AUC-PR/AUC-ROC on the test and
        precision/recall/F1 at the validation-chosen alert budget.
    """
    rows = []
    for name, pipe in finalists.items():
        proba_test = pipe.predict_proba(X_test)[:, 1]
        rate = alert_rates[name]
        at_thr = threshold_metrics(
            y_test, proba_test, threshold_for_alert_rate(proba_test, rate)
        )
        rows.append(
            {
                "modelo": name,
                "auc_pr_teste": average_precision_score(y_test, proba_test),
                "auc_roc_teste": roc_auc_score(y_test, proba_test),
                "orçamento_alertas": rate,
                "precisão_teste": at_thr["precision"],
                "recall_teste": at_thr["recall"],
                "f1_teste": at_thr["f1"],
            }
        )
    return pd.DataFrame(rows).set_index("modelo")


def monthly_stratification(
    test: pd.DataFrame, proba: np.ndarray, target: str = "is_late"
) -> pd.DataFrame:
    """Champion metrics per test month, with the month's prevalence beside.

    AUC-PR depends on prevalence, so comparing months mixes model quality
    with period difficulty; reporting prevalence on the same row (and
    AUC-ROC, which is prevalence-free) keeps the reading honest.

    Args:
        test: Test DataFrame with ``purchase_year_month`` and the target.
        proba: Champion probabilities aligned with ``test``.
        target: Target column name.

    Returns:
        One row per month: volume, prevalence, AUC-ROC, AUC-PR.
    """
    frame = test[["purchase_year_month", target]].copy()
    frame["proba"] = proba
    rows = []
    for month, grp in frame.groupby("purchase_year_month"):
        rows.append(
            {
                "mês": month,
                "pedidos": len(grp),
                "taxa_atraso": grp[target].mean(),
                "auc_roc": roc_auc_score(grp[target], grp["proba"]),
                "auc_pr": average_precision_score(grp[target], grp["proba"]),
            }
        )
    return pd.DataFrame(rows).set_index("mês")


def business_scenario(
    y_true: pd.Series, proba: np.ndarray, threshold: float, n_months: int
) -> pd.DataFrame:
    """Translate a threshold into operational counts (totals and per month).

    Args:
        y_true: Test target.
        proba: Champion probabilities on the test.
        threshold: Validation-chosen decision threshold.
        n_months: Number of months in the test window (for monthly scale).

    Returns:
        Two-row table: totals over the test window and per-month averages.
    """
    alert = proba >= threshold
    notified = int(alert.sum())
    captured = int((alert & (y_true == 1)).sum())
    false_alarms = notified - captured
    missed = int(((~alert) & (y_true == 1)).sum())
    total = {
        "clientes notificados": notified,
        "atrasos capturados": captured,
        "falsos alarmes": false_alarms,
        "atrasos não capturados": missed,
    }
    per_month = {k: round(v / n_months) for k, v in total.items()}
    out = pd.DataFrame(
        [total, per_month], index=["teste completo (6 meses)", "por mês típico"]
    )
    out["% dos atrasos capturados"] = [
        f"{captured / (captured + missed):.0%}",
        f"{captured / (captured + missed):.0%}",
    ]
    return out
