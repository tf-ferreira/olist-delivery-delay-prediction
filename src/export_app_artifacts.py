"""Export the small derived artifacts consumed by the Streamlit app.

The public repo carries no raw or processed Olist data (gitignored and not
redistributable), so the deployed app cannot rebuild the master table. This
script exports ONLY small derived artifacts to ``app/artifacts/`` (committed):
the champion pipeline, per-order test predictions (what powers the live alert
budget slider), EDA aggregates, the operating menu and simulator defaults.

Run with:

    python -m src.export_app_artifacts
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve

from src.evaluate import pick_operating_points
from src.features import build_seller_history_features
from src.train import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_model_grid,
    describe_params,
    temporal_split,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "app" / "artifacts"
CUT, TEST_END, VAL_START = "2018-03-01", "2018-09-01", "2018-01-01"
CHAMPION_PARAMS = "C=0.1"


def rate_by(df: pd.DataFrame, key, min_n: int = 0) -> pd.DataFrame:
    out = (
        df.groupby(key, observed=True)[TARGET].agg(n="size", rate="mean").reset_index()
    )
    return out[out["n"] >= min_n]


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    master = pd.read_parquet(ROOT / "data/processed/master_table.parquet")
    items = pd.read_csv(ROOT / "data/raw/olist_order_items_dataset.csv")
    master = master.merge(
        build_seller_history_features(master, items), on="order_id", how="left"
    )

    pop = master[master["order_purchase_timestamp"] < pd.Timestamp(TEST_END)]
    train, test = temporal_split(pop, CUT, TEST_END)
    core, val = temporal_split(train, VAL_START, CUT)

    # Champion (full-train refit) + per-order test predictions for the slider.
    champion = joblib.load(ROOT / "data/processed/models/regressao_logistica.joblib")
    joblib.dump(champion, ARTIFACTS / "champion.joblib")
    proba_test = champion.predict_proba(test[FEATURES])[:, 1]
    test.assign(proba=proba_test)[
        ["order_id", "purchase_year_month", TARGET, "proba"]
    ].to_parquet(ARTIFACTS / "test_predictions.parquet", index=False)

    # Operating menu chosen out-of-sample: core-fitted champion on validation.
    spw = float((core[TARGET] == 0).sum() / (core[TARGET] == 1).sum())
    core_champion = [
        p
        for p in build_model_grid(spw)["Regressão Logística"]
        if describe_params(p) == CHAMPION_PARAMS
    ][0]
    core_champion.fit(core[FEATURES], core[TARGET])
    proba_val = core_champion.predict_proba(val[FEATURES])[:, 1]
    ops = pick_operating_points(val[TARGET], proba_val)
    ops.reset_index().to_parquet(ARTIFACTS / "operating_points.parquet", index=False)

    precision, recall, _ = precision_recall_curve(val[TARGET], proba_val)
    step = max(1, len(precision) // 600)
    pd.DataFrame({"precision": precision[::step], "recall": recall[::step]}).to_parquet(
        ARTIFACTS / "pr_curve_val.parquet", index=False
    )

    # EDA aggregates for pages 1 and 2.
    rate_by(master, "purchase_year_month", min_n=100).to_parquet(
        ARTIFACTS / "monthly_rate.parquet", index=False
    )
    rate_by(master, "customer_state").to_parquet(
        ARTIFACTS / "uf_rate.parquet", index=False
    )
    for col, fname, fmt in (
        ("distance_km", "dist_rate.parquet", "{:.0f}"),
        ("promised_window_days", "window_rate.parquet", "{:.0f}"),
    ):
        bins = pd.qcut(master[col], 5, duplicates="drop")
        agg = rate_by(master.assign(_bin=bins), "_bin")
        agg["_bin"] = [
            f"{fmt.format(iv.left)}–{fmt.format(iv.right)}" for iv in agg["_bin"]
        ]
        agg.rename(columns={"_bin": "faixa"}).to_parquet(ARTIFACTS / fname, index=False)

    shutil.copy(
        ROOT / "data/processed/seller_segments.parquet", ARTIFACTS / "personas.parquet"
    )

    # Simulator defaults: train-only medians/modes (leakage discipline even here).
    defaults = {
        "medians": {c: float(train[c].median()) for c in NUMERIC_FEATURES},
        "modes": {c: str(train[c].mode().iloc[0]) for c in CATEGORICAL_FEATURES},
        "states": sorted(train["customer_state"].unique().tolist()),
        "categories": train["main_category"].value_counts().head(30).index.tolist(),
        "payment_types": sorted(
            train["payment_type_primary"].dropna().unique().tolist()
        ),
    }
    (ARTIFACTS / "simulator_defaults.json").write_text(
        json.dumps(defaults, ensure_ascii=False)
    )

    meta = {
        "cut": CUT,
        "test_end": TEST_END,
        "n_test": int(len(test)),
        "n_test_months": 6,
        "test_prevalence": float(test[TARGET].mean()),
        "population": int(len(master)),
        "global_rate": float(master[TARGET].mean()),
        "default_alert_rate": float(ops.loc["equilibrado", "alert_rate_val"]),
        "named_points": {
            name: float(row["alert_rate_val"]) for name, row in ops.iterrows()
        },
        "champion": "Regressão Logística (C=0,1)",
        "features": FEATURES,
    }
    (ARTIFACTS / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    total = sum(f.stat().st_size for f in ARTIFACTS.iterdir())
    print(
        f"artifacts em {ARTIFACTS.relative_to(ROOT)}: {len(list(ARTIFACTS.iterdir()))} arquivos, {total / 1e6:.2f} MB"
    )


if __name__ == "__main__":
    main()
