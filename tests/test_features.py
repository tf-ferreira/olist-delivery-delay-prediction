"""Tests for the hermetic seller-history features.

These tests encode the anti-leakage CONTRACT of the project's central
feature. Each one describes a scenario a rigorous reviewer would probe:
outcomes must not be visible before delivery, posting times become visible
at carrier hand-off (an earlier clock), cold start falls back to the global
history with a flag, and multi-seller orders take the worst seller.
"""

import pandas as pd
import pytest

from src.features import build_seller_history_features


def _master(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in (
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
    ):
        df[col] = pd.to_datetime(df[col])
    return df


def _items(mapping: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"order_id": order, "seller_id": seller}
            for order, sellers in mapping.items()
            for seller in sellers
        ]
    )


@pytest.fixture()
def toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Timeline (seller S1): O1 bought 01-01, posted 01-03, delivered 01-10 (late).

    O2 is bought 01-05, BETWEEN O1's posting and O1's delivery: it may see
    O1's posting time but NOT O1's outcome. O3 is bought 02-01, after both
    deliveries. S2 is a one-order seller whose order is always late; S3
    never appears before, testing pure cold start.
    """
    master = _master(
        [
            dict(
                order_id="O1",
                order_purchase_timestamp="2017-01-01",
                order_delivered_carrier_date="2017-01-03",
                order_delivered_customer_date="2017-01-10",
                is_late=1,
            ),
            dict(
                order_id="O2",
                order_purchase_timestamp="2017-01-05",
                order_delivered_carrier_date="2017-01-07",
                order_delivered_customer_date="2017-01-20",
                is_late=0,
            ),
            dict(
                order_id="O3",
                order_purchase_timestamp="2017-02-01",
                order_delivered_carrier_date="2017-02-10",
                order_delivered_customer_date="2017-02-15",
                is_late=0,
            ),
            dict(
                order_id="OS2",
                order_purchase_timestamp="2017-01-02",
                order_delivered_carrier_date="2017-01-12",
                order_delivered_customer_date="2017-01-15",
                is_late=1,
            ),
            dict(
                order_id="O4",
                order_purchase_timestamp="2017-03-01",
                order_delivered_carrier_date="2017-03-05",
                order_delivered_customer_date="2017-03-09",
                is_late=0,
            ),
            dict(
                order_id="O5",
                order_purchase_timestamp="2017-03-10",
                order_delivered_carrier_date="2017-03-12",
                order_delivered_customer_date="2017-03-20",
                is_late=0,
            ),
        ]
    )
    items = _items(
        {
            "O1": ["S1"],
            "O2": ["S1"],
            "O3": ["S1"],
            "OS2": ["S2"],
            "O4": ["S1", "S2"],  # multi-seller
            "O5": ["S3"],  # brand-new seller
        }
    )
    return master, items


def _row(result: pd.DataFrame, order_id: str) -> pd.Series:
    return result.set_index("order_id").loc[order_id]


def test_outcome_invisible_before_delivery(toy):
    """O2 (bought 01-05) must NOT see O1's outcome (delivered 01-10)."""
    master, items = toy
    row = _row(build_seller_history_features(master, items), "O2")
    # No delivered order existed anywhere before 01-05, so even the global
    # fallback is empty: NaN + flag is the only honest answer.
    assert row["seller_no_history"] == 1
    assert pd.isna(row["seller_late_rate_hist"])


def test_posting_clock_runs_earlier_than_outcome_clock(toy):
    """O2 may see O1's POSTING (carrier 01-03) even though O1's outcome is unknown."""
    master, items = toy
    row = _row(build_seller_history_features(master, items), "O2")
    assert row["seller_posting_days_hist"] == pytest.approx(2.0)  # O1: 01-01 -> 01-03


def test_history_uses_only_prior_deliveries(toy):
    """O3 (bought 02-01) sees O1 (late) and O2 (on time): rate = 0.5."""
    master, items = toy
    row = _row(build_seller_history_features(master, items), "O3")
    assert row["seller_late_rate_hist"] == pytest.approx(0.5)
    assert row["seller_no_history"] == 0


def test_multi_seller_takes_the_worst_seller(toy):
    """O4 mixes S1 (rate 1/3) and S2 (rate 1.0): weakest link wins."""
    master, items = toy
    row = _row(build_seller_history_features(master, items), "O4")
    assert row["seller_late_rate_hist"] == pytest.approx(1.0)


def test_cold_start_falls_back_to_global_history(toy):
    """O5's seller S3 has no past: value = global late rate before 03-10, flagged."""
    master, items = toy
    row = _row(build_seller_history_features(master, items), "O5")
    # Delivered before 03-10: O1 (late), OS2 (late), O2, O3, O4 (on time) -> 2/5.
    assert row["seller_late_rate_hist"] == pytest.approx(2 / 5)
    assert row["seller_no_history"] == 1


def test_deterministic_output(toy):
    """Same input, same output, regardless of input row order."""
    master, items = toy
    a = build_seller_history_features(master, items)
    b = build_seller_history_features(
        master.sample(frac=1, random_state=0), items.sample(frac=1, random_state=1)
    )
    pd.testing.assert_frame_equal(
        a.sort_values("order_id").reset_index(drop=True),
        b.sort_values("order_id").reset_index(drop=True),
    )
