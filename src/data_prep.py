"""Build the order-level master table for the delivery delay project.

This module is the single source of truth for the master table consumed by
the EDA notebook, the modeling notebook and the Streamlit app. Run it with:

    python -m src.data_prep

Design decisions enforced here (see notebook 01 for the full audit):

* Population: orders with status "delivered" and a non-null delivery date.
* Target: ``is_late`` compares DATES, not timestamps, because the estimated
  delivery date is always recorded at midnight.
* Geolocation is filtered to Brazil's bounding box and reduced to one
  centroid per zip prefix BEFORE any join, otherwise merges multiply rows.
* Every merge is audited: the row count of the order-level table must never
  change. A violation raises immediately instead of silently corrupting data.
* Missing values are kept as NaN plus an explicit flag column instead of
  being imputed here: imputation statistics must be learned on the training
  split only (computing them on the full table would leak the test period).
* Post-purchase timestamps (approval, carrier hand-off, delivery) are CARRIED
  in the table for descriptive analysis but are banned as model features by
  the leakage contract in notebook 01; feature selection happens at modeling.
"""

from __future__ import annotations

import logging

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MASTER_TABLE_PATH = PROCESSED_DIR / "master_table.parquet"

RAW_FILES = {
    "orders": "olist_orders_dataset.csv",
    "order_items": "olist_order_items_dataset.csv",
    "order_payments": "olist_order_payments_dataset.csv",
    "customers": "olist_customers_dataset.csv",
    "sellers": "olist_sellers_dataset.csv",
    "products": "olist_products_dataset.csv",
    "geolocation": "olist_geolocation_dataset.csv",
}

ORDER_TIMESTAMP_COLS = [
    "order_purchase_timestamp",
    "order_approved_at",
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_estimated_delivery_date",
]

# Brazil bounding box (includes oceanic islands); measured in notebook 01:
# 31 geolocation rows fall outside and would poison prefix centroids.
BRAZIL_LAT_BOUNDS = (-33.75, 5.30)
BRAZIL_LNG_BOUNDS = (-74.00, -28.80)
EARTH_RADIUS_KM = 6371.0

logger = logging.getLogger(__name__)


def load_raw_tables(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """Load the raw CSVs needed for the master table.

    Reviews are not loaded: they are 100% leakage as features and their
    descriptive use in the EDA reads the CSV directly.

    Args:
        raw_dir: Directory containing the Kaggle CSVs.

    Returns:
        Mapping of table name to raw DataFrame, with order timestamps parsed.
    """
    tables = {name: pd.read_csv(raw_dir / fname) for name, fname in RAW_FILES.items()}
    for col in ORDER_TIMESTAMP_COLS:
        tables["orders"][col] = pd.to_datetime(tables["orders"][col])
    tables["order_items"]["shipping_limit_date"] = pd.to_datetime(
        tables["order_items"]["shipping_limit_date"]
    )
    return tables


def _audit_rows(df: pd.DataFrame, expected: int, step: str) -> pd.DataFrame:
    """Fail loudly if a merge changed the order-level row count."""
    if len(df) != expected:
        raise AssertionError(f"{step}: row count changed from {expected} to {len(df)}")
    logger.info("%-52s %d rows (unchanged)", step, len(df))
    return df


def build_order_population(orders: pd.DataFrame) -> pd.DataFrame:
    """Filter the modeling population and derive the target.

    The target compares dates, not timestamps: the estimated delivery date is
    always midnight, so a timestamp comparison would mislabel orders delivered
    later the same promised day (1,292 orders, measured in notebook 01).

    Args:
        orders: Raw orders table with parsed timestamps.

    Returns:
        One row per delivered order with a non-null delivery date, including
        ``is_late`` and purchase-time helper columns.
    """
    population = orders[orders["order_status"] == "delivered"].copy()
    n_no_date = int(population["order_delivered_customer_date"].isna().sum())
    population = population.dropna(subset=["order_delivered_customer_date"])
    logger.info(
        "population: %d delivered orders (%d dropped for null delivery date)",
        len(population),
        n_no_date,
    )

    delivered_date = population["order_delivered_customer_date"].dt.normalize()
    population["is_late"] = (
        delivered_date > population["order_estimated_delivery_date"]
    ).astype("int8")

    purchase = population["order_purchase_timestamp"]
    population["promised_window_days"] = (
        population["order_estimated_delivery_date"] - purchase.dt.normalize()
    ).dt.days
    population["purchase_month"] = purchase.dt.month.astype("int8")
    population["purchase_year_month"] = purchase.dt.strftime("%Y-%m")
    return population.drop(columns=["order_status"])


def build_geolocation_centroids(geolocation: pd.DataFrame) -> pd.DataFrame:
    """Reduce geolocation to one median centroid per zip prefix.

    The bounding-box filter comes BEFORE the median: out-of-country points
    (e.g. lat +45) would drag centroids even under a median for prefixes
    with few rows.

    Args:
        geolocation: Raw geolocation table (~1M rows, no key).

    Returns:
        DataFrame indexed by zip prefix with ``lat`` and ``lng`` columns.
    """
    inside = geolocation["geolocation_lat"].between(*BRAZIL_LAT_BOUNDS) & geolocation[
        "geolocation_lng"
    ].between(*BRAZIL_LNG_BOUNDS)
    logger.info(
        "geolocation: dropped %d rows outside Brazil's bounding box",
        int((~inside).sum()),
    )
    centroids = (
        geolocation.loc[inside]
        .groupby("geolocation_zip_code_prefix")[["geolocation_lat", "geolocation_lng"]]
        .median()
        .rename(columns={"geolocation_lat": "lat", "geolocation_lng": "lng"})
    )
    logger.info("geolocation: %d prefix centroids", len(centroids))
    return centroids


def haversine_km(
    lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series
) -> np.ndarray:
    """Vectorized great-circle distance in kilometers.

    Args:
        lat1: Latitudes of the first points, in degrees.
        lng1: Longitudes of the first points, in degrees.
        lat2: Latitudes of the second points, in degrees.
        lng2: Longitudes of the second points, in degrees.

    Returns:
        Array of distances in kilometers (NaN where any input is NaN).
    """
    lat1_r, lng1_r, lat2_r, lng2_r = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2_r - lat1_r
    dlng = lng2_r - lng1_r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def aggregate_items(order_items: pd.DataFrame, products: pd.DataFrame) -> pd.DataFrame:
    """Aggregate order items to one row per order.

    The order category and main seller come from the highest-priced item
    (the item that dominates the customer's experience). Weight/volume sums
    stay NaN when any component is missing (2 products in the catalog),
    flagged instead of silently summing partial values.

    Args:
        order_items: Raw order items table.
        products: Raw products table.

    Returns:
        One row per order with item aggregates.
    """
    products = products.copy()
    products["product_category_name"] = products["product_category_name"].fillna(
        "desconhecida"
    )
    products["volume_cm3"] = (
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"]
    )

    items = order_items.merge(
        products[
            ["product_id", "product_category_name", "product_weight_g", "volume_cm3"]
        ],
        on="product_id",
        how="left",
    )

    grp = items.groupby("order_id")
    agg = grp.agg(
        n_items=("order_item_id", "size"),
        n_sellers=("seller_id", "nunique"),
        total_price=("price", "sum"),
        total_freight=("freight_value", "sum"),
        total_weight_g=("product_weight_g", "sum"),
        total_volume_cm3=("volume_cm3", "sum"),
    )
    # Propagate NaN when any item lacks weight/dimensions, and flag it.
    complete_weight = grp["product_weight_g"].count() == grp.size()
    agg["total_weight_g"] = agg["total_weight_g"].where(complete_weight)
    agg["total_volume_cm3"] = agg["total_volume_cm3"].where(complete_weight)
    agg["weight_missing"] = (~complete_weight).astype("int8")

    agg["freight_price_ratio"] = (agg["total_freight"] / agg["total_price"]).where(
        agg["total_price"] > 0
    )

    top_item = items.sort_values(
        ["order_id", "price", "order_item_id"], ascending=[True, False, True]
    ).drop_duplicates("order_id")
    agg = agg.join(
        top_item.set_index("order_id")[["product_category_name", "seller_id"]].rename(
            columns={
                "product_category_name": "main_category",
                "seller_id": "main_seller_id",
            }
        )
    )
    return agg.reset_index()


def aggregate_payments(order_payments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate payments to one row per order.

    The primary payment type is the one carrying the highest value, which
    represents the order for the rare multi-method checkouts.

    Args:
        order_payments: Raw payments table.

    Returns:
        One row per order with payment aggregates.
    """
    agg = order_payments.groupby("order_id").agg(
        payment_total=("payment_value", "sum"),
        payment_installments_max=("payment_installments", "max"),
    )
    primary = order_payments.sort_values(
        ["order_id", "payment_value"], ascending=[True, False]
    ).drop_duplicates("order_id")
    agg["payment_type_primary"] = primary.set_index("order_id")["payment_type"]
    return agg.reset_index()


def compute_order_distances(
    order_items: pd.DataFrame,
    population: pd.DataFrame,
    customers: pd.DataFrame,
    sellers: pd.DataFrame,
    centroids: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the seller→customer distance for each order.

    Multi-seller orders take the MAXIMUM distance across sellers: the order
    is only complete when the last item arrives, so the farthest seller
    dominates the delivery risk (weakest-link rule, consistent with the
    seller risk aggregation). Distances stay NaN when either endpoint's zip
    prefix has no geolocation coverage, with an explicit flag.

    Args:
        order_items: Raw order items table.
        population: Order population with ``customer_id``.
        customers: Raw customers table.
        sellers: Raw sellers table.
        centroids: Zip-prefix centroids from :func:`build_geolocation_centroids`.

    Returns:
        One row per order with ``distance_km`` and ``distance_missing``.
    """
    pairs = order_items[["order_id", "seller_id"]].drop_duplicates()
    pairs = pairs[pairs["order_id"].isin(population["order_id"])]

    seller_coords = sellers.merge(
        centroids, left_on="seller_zip_code_prefix", right_index=True, how="left"
    )[["seller_id", "lat", "lng"]].rename(
        columns={"lat": "seller_lat", "lng": "seller_lng"}
    )

    customer_coords = population[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_zip_code_prefix"]],
        on="customer_id",
        how="left",
    )
    customer_coords = customer_coords.merge(
        centroids, left_on="customer_zip_code_prefix", right_index=True, how="left"
    )[["order_id", "lat", "lng"]].rename(
        columns={"lat": "customer_lat", "lng": "customer_lng"}
    )

    pairs = pairs.merge(seller_coords, on="seller_id", how="left")
    pairs = pairs.merge(customer_coords, on="order_id", how="left")
    pairs["distance_km"] = haversine_km(
        pairs["seller_lat"],
        pairs["seller_lng"],
        pairs["customer_lat"],
        pairs["customer_lng"],
    )

    distances = pairs.groupby("order_id")["distance_km"].max().to_frame()
    distances["distance_missing"] = distances["distance_km"].isna().astype("int8")
    return distances.reset_index()


def build_master_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Assemble the order-level master table with audited joins.

    Args:
        tables: Raw tables from :func:`load_raw_tables`.

    Returns:
        One row per order in the population, with target, purchase-time
        features and carried (banned) post-purchase timestamps.
    """
    population = build_order_population(tables["orders"])
    n = len(population)

    centroids = build_geolocation_centroids(tables["geolocation"])

    master = population.merge(
        tables["customers"][
            [
                "customer_id",
                "customer_unique_id",
                "customer_zip_code_prefix",
                "customer_city",
                "customer_state",
            ]
        ],
        on="customer_id",
        how="left",
    )
    _audit_rows(master, n, "join customers (1:1)")

    master = master.merge(
        aggregate_items(tables["order_items"], tables["products"]),
        on="order_id",
        how="left",
    )
    _audit_rows(master, n, "join item aggregates (1:1 by construction)")

    master = master.merge(
        aggregate_payments(tables["order_payments"]), on="order_id", how="left"
    )
    _audit_rows(master, n, "join payment aggregates (1:1 by construction)")
    master["payment_missing"] = master["payment_total"].isna().astype("int8")

    master = master.merge(
        compute_order_distances(
            tables["order_items"],
            population,
            tables["customers"],
            tables["sellers"],
            centroids,
        ),
        on="order_id",
        how="left",
    )
    _audit_rows(master, n, "join seller→customer distances (1:1 by construction)")

    return master


def summarize_nulls(master: pd.DataFrame) -> pd.DataFrame:
    """Report null counts with the documented reason for each affected column.

    Args:
        master: Assembled master table.

    Returns:
        DataFrame with null counts and reasons, for logging and notebooks.
    """
    reasons = {
        "order_approved_at": "carried post-purchase timestamp (EDA only, banned as feature)",
        "order_delivered_carrier_date": "carried post-purchase timestamp (EDA only, banned as feature)",
        "total_weight_g": "2 catalog products lack weight/dimensions; flagged in weight_missing",
        "total_volume_cm3": "2 catalog products lack weight/dimensions; flagged in weight_missing",
        "payment_total": "1 order has no payment record; flagged in payment_missing",
        "payment_installments_max": "1 order has no payment record; flagged in payment_missing",
        "payment_type_primary": "1 order has no payment record; flagged in payment_missing",
        "distance_km": "zip prefixes without geolocation coverage; flagged in distance_missing",
    }
    nulls = master.isna().sum()
    nulls = nulls[nulls > 0]
    return pd.DataFrame(
        {
            "nulls": nulls,
            "reason": [
                reasons.get(col, "UNEXPECTED, investigate") for col in nulls.index
            ],
        }
    )


def main() -> None:
    """Build the master table, log the audit and write the parquet."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    tables = load_raw_tables()
    master = build_master_table(tables)

    null_summary = summarize_nulls(master)
    logger.info("null audit:\n%s", null_summary.to_string())
    unexpected = null_summary[null_summary["reason"].eq("UNEXPECTED, investigate")]
    if not unexpected.empty:
        raise AssertionError(f"unexpected nulls: {list(unexpected.index)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    master.to_parquet(MASTER_TABLE_PATH, index=False)
    logger.info(
        "written %s: %d rows, %d columns, late rate %.2f%%",
        MASTER_TABLE_PATH.relative_to(ROOT),
        len(master),
        master.shape[1],
        100 * master["is_late"].mean(),
    )


if __name__ == "__main__":
    main()
