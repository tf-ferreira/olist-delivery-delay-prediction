"""Leakage-proof seller history features (the technical core of the project).

Two features summarize each seller's past at the exact moment a new order is
placed:

* ``seller_late_rate_hist``: share of the seller's PAST orders delivered late.
* ``seller_posting_days_hist``: seller's average days between purchase and
  hand-off to the carrier, over PAST orders.

The construction is *hermetic*: an order purchased at time ``t`` may only see
information that PHYSICALLY EXISTED at ``t``. A naive ``expanding().shift(1)``
over purchase order is NOT enough, because the outcome of the previous order
(late or not) only comes into existence when that order is DELIVERED, about
ten days after its purchase. An order placed two days after the previous one
would "know" an outcome nobody knew yet.

Each piece of information therefore enters the history at the moment it is
born, through its own availability clock:

* the late/on-time OUTCOME becomes known at ``order_delivered_customer_date``;
* the POSTING TIME becomes known at ``order_delivered_carrier_date`` (the
  moment the package reaches the carrier, before delivery).

Cold start: when a seller has no delivered order before ``t``, the feature
falls back to the GLOBAL history (same hermetic rule, all sellers pooled) and
the order is flagged with ``seller_no_history = 1``. At the very start of the
dataset even the global history is empty; the value stays NaN and is handled
by the train-only median imputer in the modeling pipeline.

Multi-seller orders take the MAXIMUM across their sellers (weakest-link rule,
consistent with the distance aggregation): the order is only complete when
the last package arrives. Note this reads ALL sellers of the order, not
``main_seller_id``, which sidesteps part of the order-vs-shipment grain
limitation documented in the project notes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mean_of_events_before(
    avail_ts: np.ndarray, values: np.ndarray, query_ts: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of the values whose availability timestamp is strictly before each query.

    This is the hermetic primitive: an event (an outcome, a posting time)
    only counts toward a query made at time ``t`` if the event's availability
    timestamp is < ``t``. Implemented with a sorted cumulative sum plus binary
    search, so each seller's history costs O(n log n) instead of O(n²).

    Args:
        avail_ts: Timestamps (int64 ns) at which each event became known.
        values: The value of each event (e.g., 0/1 outcome, posting days).
        query_ts: Timestamps (int64 ns) of the queries (order purchases).

    Returns:
        Tuple ``(means, counts)`` aligned with ``query_ts``; mean is NaN
        where no event was available yet.
    """
    order = np.argsort(avail_ts, kind="stable")
    avail_sorted = avail_ts[order]
    cumsum = np.cumsum(values[order], dtype="float64")
    # side="left" makes the comparison strict: an event available exactly at
    # the purchase instant is NOT visible (it did not precede the purchase).
    idx = np.searchsorted(avail_sorted, query_ts, side="left")
    counts = idx.astype("int64")
    sums = np.where(idx > 0, cumsum[np.maximum(idx - 1, 0)], 0.0)
    means = np.divide(
        sums, counts, out=np.full(len(query_ts), np.nan), where=counts > 0
    )
    return means, counts


def build_seller_history_features(
    master: pd.DataFrame, order_items: pd.DataFrame
) -> pd.DataFrame:
    """Build the two hermetic seller-history features, one row per order.

    Args:
        master: Order-level table with ``order_id``, ``order_purchase_timestamp``,
            ``order_delivered_customer_date``, ``order_delivered_carrier_date``
            and ``is_late``.
        order_items: Raw items table with ``order_id`` and ``seller_id`` (used
            to map orders to ALL their sellers, not only the main one).

    Returns:
        DataFrame with ``order_id``, ``seller_late_rate_hist``,
        ``seller_posting_days_hist`` and ``seller_no_history`` (int8).
    """
    base = master[
        [
            "order_id",
            "order_purchase_timestamp",
            "order_delivered_customer_date",
            "order_delivered_carrier_date",
            "is_late",
        ]
    ].copy()
    base["posting_days"] = (
        base["order_delivered_carrier_date"] - base["order_purchase_timestamp"]
    ).dt.total_seconds() / 86400

    # One row per (order, seller): the feature is computed per seller and
    # aggregated back to the order with the weakest-link rule.
    pairs = (
        order_items[["order_id", "seller_id"]]
        .drop_duplicates()
        .merge(base, on="order_id", how="inner")
        .sort_values(["seller_id", "order_purchase_timestamp", "order_id"])
        .reset_index(drop=True)
    )

    purchase_ns = pairs["order_purchase_timestamp"].astype("int64").to_numpy()

    # --- availability clock 1: outcomes exist at DELIVERY time -------------
    outcome_avail_ns = pairs["order_delivered_customer_date"].astype("int64").to_numpy()
    outcome_value = pairs["is_late"].to_numpy(dtype="float64")

    # --- availability clock 2: posting times exist at CARRIER time ---------
    posting_known = pairs["posting_days"].notna().to_numpy()

    late_hist = np.full(len(pairs), np.nan)
    late_count = np.zeros(len(pairs), dtype="int64")
    post_hist = np.full(len(pairs), np.nan)

    # Per-seller hermetic histories. groupby(sort=False) keeps the temporal
    # sort applied above, which makes the whole computation deterministic.
    for _, idx in pairs.groupby("seller_id", sort=False).indices.items():
        q = purchase_ns[idx]
        late_hist[idx], late_count[idx] = _mean_of_events_before(
            outcome_avail_ns[idx], outcome_value[idx], q
        )
        known = posting_known[idx]
        if known.any():
            sub = idx[known]
            post_hist[idx], _ = _mean_of_events_before(
                pairs["order_delivered_carrier_date"].astype("int64").to_numpy()[sub],
                pairs["posting_days"].to_numpy(dtype="float64")[sub],
                q,
            )

    # Global hermetic fallback (cold start): the same rule, all sellers
    # pooled. Deduplicated to one event per ORDER so multi-seller orders do
    # not weigh twice in the global average.
    order_events = base.dropna(subset=["order_delivered_customer_date"])
    g_late, _ = _mean_of_events_before(
        order_events["order_delivered_customer_date"].astype("int64").to_numpy(),
        order_events["is_late"].to_numpy(dtype="float64"),
        purchase_ns,
    )
    post_events = base.dropna(subset=["posting_days"])
    g_post, _ = _mean_of_events_before(
        post_events["order_delivered_carrier_date"].astype("int64").to_numpy(),
        post_events["posting_days"].to_numpy(dtype="float64"),
        purchase_ns,
    )

    no_history = late_count == 0
    pairs["late_rate_hist"] = np.where(no_history, g_late, late_hist)
    pairs["posting_days_hist"] = np.where(np.isnan(post_hist), g_post, post_hist)
    pairs["no_history"] = no_history

    # Weakest link: the order's risk is its riskiest seller.
    per_order = pairs.groupby("order_id", sort=False).agg(
        seller_late_rate_hist=("late_rate_hist", "max"),
        seller_posting_days_hist=("posting_days_hist", "max"),
        seller_no_history=("no_history", "any"),
    )
    per_order["seller_no_history"] = per_order["seller_no_history"].astype("int8")
    return per_order.reset_index()
