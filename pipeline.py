"""
NYC TLC ETA Prediction — Full Pipeline
========================================

Run this script to execute the complete end-to-end pipeline:

  Step 1 — Data Cleaning
  Step 2 — Data Splitting + Subsampling
  Step 3 — Experiment A: Baseline        (raw features only, no engineering)
  Step 4 — Experiment B: Full Engineering (all feature creation, transformation, store)
  Step 5 — Head-to-head comparison        (same models, same data, features are the only difference)
  Step 6 — Feature Importance             (champion engineered model)

The side-by-side comparison in Steps 3–5 is the core teaching moment:
identical models trained on identical data differ only in their feature set.
Any performance gap is directly attributable to feature engineering.
"""

import joblib
from pathlib import Path

from src.cleaning   import clean_parquet
from src.splitting  import split_train_test, subsample_splits
from src.features   import (
    run_baseline_pipeline,
    run_feature_pipeline,
    TARGET_COL,
)
from src.models     import train_all_models, load_model
from src.evaluation import evaluate_all_models, select_champion, plot_feature_importance


# ── Path configuration ────────────────────────────────────────────────────────

RAW_CLEAN_PARQUET  = "data/processed/yellow_tripdata_2024-01_clean.parquet"
CLEANED_PARQUET    = "data/processed/yellow_tripdata_2024-01_cleaned.parquet"
PROCESSED_DIR      = "data/processed"
MODEL_DIR_BASELINE = "models/baseline"
MODEL_DIR_ENGINEERED = "models/engineered"
PLOTS_DIR          = "outputs/plots"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _print_header(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def demo_zone_demand_concept(df):
    """
    Teaching demo: show one representative trip from each of five different
    pickup zones, sorted chronologically.

    The goal is to set up the aggregation concept:
      "For each (PULocationID, hour_of_week) combination, how many trips
       historically depart from that zone in that time window?"
    Seeing individual rows first makes it concrete before we group and average.
    """
    display_cols = ["tpep_pickup_datetime", "PULocationID", "passenger_count", "trip_distance"]

    # 1. Pick 5 random unique pickup locations
    # 2. For each, sample 5 random rides
    # 3. Sort chronologically within each location
    random_zones = df["PULocationID"].drop_duplicates().sample(5)

    def add_zone_average(grp):
        grp = grp.sample(5).sort_values("tpep_pickup_datetime").copy()
        grp["zone_average"] = grp["trip_distance"].iloc[:4].mean()
        return grp

    sample = (
        df[df["PULocationID"].isin(random_zones)]
        .groupby("PULocationID", sort=False)
        .apply(add_zone_average)
        .reset_index()
        [display_cols + ["zone_average"]]
    )

    print("\n  One trip per zone (5 zones, chronological order):")
    print(sample.to_string(index=False))
    print(
        "\n  -> Each row is a single trip. "
        "By grouping on PULocationID + hour_of_week and counting rows, "
        "we get the average number of departures per zone per hour — "
        "zone-level demand — which we can attach as a feature to every trip."
    )


def _comparison_table(baseline_results, engineered_results):
    """
    Print a side-by-side comparison table and return the merged DataFrame.
    Improvement is the percentage reduction in MAE achieved by engineering.
    """
    merged = baseline_results[["model", "mae", "rmse"]].merge(
        engineered_results[["model", "mae", "rmse"]],
        on="model",
        suffixes=("_baseline", "_engineered"),
    )
    merged["mae_improvement_%"] = (
        (merged["mae_baseline"] - merged["mae_engineered"]) / merged["mae_baseline"] * 100
    ).round(1)

    print(f"\n  {'Model':<25} {'Baseline MAE':>13} {'Engineered MAE':>15} {'Improvement':>12}")
    print("  " + "-" * 68)
    for _, row in merged.iterrows():
        print(
            f"  {row['model']:<25} "
            f"{row['mae_baseline']:>13.2f} "
            f"{row['mae_engineered']:>15.2f} "
            f"{row['mae_improvement_%']:>11.1f}%"
        )
    return merged


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline():

    # ── Step 1: Data Cleaning ──────────────────────────────────────────────────
    _print_header("STEP 1 — Data Cleaning")
    df_clean = clean_parquet(RAW_CLEAN_PARQUET, CLEANED_PARQUET)
    demo_zone_demand_concept(df_clean)

    # ── Step 2: Data Splitting + Subsampling ───────────────────────────────────
    _print_header("STEP 2 — Data Splitting + Subsampling")
    train_path, test_path = split_train_test(CLEANED_PARQUET, PROCESSED_DIR)
    print("\n  Subsampling splits ...")
    train_raw, test_raw = subsample_splits(train_path, test_path)

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT A — Baseline (raw features only)
    #
    # The same models are trained on the same rows using ONLY the four raw,
    # non-leaky columns: passenger_count, trip_distance,
    # PULocationID, DOLocationID.
    # No time features. No interaction terms.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 3 — Experiment A: Baseline (Raw Features Only)")

    baseline_train, baseline_scaler = run_baseline_pipeline(train_raw, is_training=True)
    X_train_base = baseline_train.drop(columns=[TARGET_COL])
    y_train_base = baseline_train[TARGET_COL]
    print(f"  Baseline feature columns ({len(X_train_base.columns)}): {X_train_base.columns.tolist()}")

    train_all_models(X_train_base, y_train_base, MODEL_DIR_BASELINE)

    baseline_test, _ = run_baseline_pipeline(test_raw, scaler=baseline_scaler, is_training=False)
    X_test_base = baseline_test.drop(columns=[TARGET_COL])
    y_test_base = baseline_test[TARGET_COL]

    print("\n  Baseline model results:")
    baseline_results = evaluate_all_models(X_test_base, y_test_base, MODEL_DIR_BASELINE)

    ## Persist scaler so it can be reloaded independently if needed
    #Path(MODEL_DIR_BASELINE).mkdir(parents=True, exist_ok=True)
    #joblib.dump(baseline_scaler, Path(MODEL_DIR_BASELINE) / "scaler.pkl")

    # ══════════════════════════════════════════════════════════════════════════
    # EXPERIMENT B — Full Feature Engineering
    #
    # Same models, same rows — but now the full feature pipeline runs:
    # temporal features, interaction terms, and cyclical encoding.
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 4 — Experiment B: Full Feature Engineering")

    eng_train, eng_scaler = run_feature_pipeline(train_raw, is_training=True)

    X_train_eng = eng_train.drop(columns=[TARGET_COL])
    y_train_eng = eng_train[TARGET_COL]
    print(f"  Engineered feature columns ({len(X_train_eng.columns)}): {X_train_eng.columns.tolist()}")

    train_all_models(X_train_eng, y_train_eng, MODEL_DIR_ENGINEERED)

    eng_test, _ = run_feature_pipeline(test_raw, scaler=eng_scaler, is_training=False)
    X_test_eng = eng_test.drop(columns=[TARGET_COL])
    y_test_eng = eng_test[TARGET_COL]

    print("\n  Engineered model results:")
    engineered_results = evaluate_all_models(X_test_eng, y_test_eng, MODEL_DIR_ENGINEERED)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5 — Head-to-head comparison
    # ══════════════════════════════════════════════════════════════════════════

    _print_header("STEP 5 — Feature Engineering Impact: Head-to-Head Comparison")
    print("  Metric: MAE (minutes).  Lower is better.")
    print("  Improvement = % reduction in MAE achieved by feature engineering.\n")
    comparison = _comparison_table(baseline_results, engineered_results)

    best_improvement = comparison["mae_improvement_%"].max()
    best_model_row   = comparison.loc[comparison["mae_improvement_%"].idxmax()]
    print(
        f"\n  Largest gain : {best_model_row['model']} "
        f"improved by {best_improvement:.1f}% with feature engineering"
    )

    # ── Step 6: Champion + Feature Importance ─────────────────────────────────
    _print_header("STEP 6 — Champion Model + Feature Importance")
    champion_name = select_champion(engineered_results, metric="mae")
    champion_model = load_model(champion_name, MODEL_DIR_ENGINEERED)

    plot_feature_importance(
        model         = champion_model,
        feature_names = X_test_eng.columns.tolist(),
        model_name    = champion_name,
        output_dir    = PLOTS_DIR,
    )


if __name__ == "__main__":
    run_pipeline()
