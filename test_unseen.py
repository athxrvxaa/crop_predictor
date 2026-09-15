"""
Evaluate the trained crop classifier on the unseen CSV without retraining.

Run with:
    python test_unseen.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from pipeline import (
    CROP_CONFIG,
    DEFAULT_CFG,
    FEATURE_COLS,
    RULES,
    _trapz,
    detect_cycles,
    split_into_years,
)


def _normalize_columns(raw: pd.DataFrame) -> pd.DataFrame:
    def _norm(col: str) -> str:
        return col.strip().lower().replace(" ", "_")

    rename_map = {}
    for col in raw.columns:
        name = _norm(col)
        if name in ("farm", "farm_id", "farmid"):
            rename_map[col] = "Farm"
        elif name in (
            "crop",
            "crop_label",
            "crop_name",
            "cropname",
            "crop_type",
            "croptype",
            "label",
            "class",
        ):
            rename_map[col] = "Crop"
        elif name in ("date", "sentinel_date", "image_date"):
            rename_map[col] = "Date"
        elif name == "ndvi":
            rename_map[col] = "NDVI"
        elif name in ("ground_truth_date", "ground_truth_month", "groundtruth"):
            rename_map[col] = "Ground_Truth_Date"

    return raw.rename(columns=rename_map)


def _prepare_unseen_data(csv_path: Path) -> pd.DataFrame:
    raw = pd.read_csv(csv_path)
    raw = _normalize_columns(raw)

    if "Farm" not in raw.columns and "Farm_ID" in raw.columns:
        raw = raw.rename(columns={"Farm_ID": "Farm"})

    if "Crop" not in raw.columns and "crop label" in raw.columns:
        raw = raw.rename(columns={"crop label": "Crop"})

    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw["Ground_Truth_Date"] = pd.to_datetime(raw["Ground_Truth_Date"], errors="coerce")
    raw["NDVI"] = pd.to_numeric(raw["NDVI"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Ground_Truth_Date", "NDVI", "Farm", "Crop"])
    raw = raw[raw["NDVI"] >= 0]

    if "Latitude" in raw.columns and "Longitude" in raw.columns and raw["Farm"].nunique() <= 1:
        coords = raw[["Latitude", "Longitude"]].drop_duplicates().reset_index(drop=True)
        coords["Farm"] = [f"F{i + 1:02d}" for i in range(len(coords))]
        raw = raw.drop(columns=["Farm"]).merge(coords, on=["Latitude", "Longitude"], how="left")

    if "Latitude" in raw.columns:
        raw = (
            raw.groupby(["Farm", "Crop", "Date"], as_index=False)
            .agg(NDVI=("NDVI", "mean"), Ground_Truth_Date=("Ground_Truth_Date", "first"))
        )

    return raw.sort_values(["Crop", "Farm", "Date"]).reset_index(drop=True)


def _validate_cycle(crop: str, row: pd.Series) -> bool:
    rule = RULES.get(crop)
    if rule is None:
        return False

    reasons = []
    if row["Sowing_Month"] not in rule["sow_months"]:
        reasons.append("sow_month")
    if row["Harvest_Month"] not in rule["harv_months"]:
        reasons.append("harv_month")
    if not (rule["min_dur"] <= row["Duration_Days"] <= rule["max_dur"]):
        reasons.append("duration")
    if (row["Peak_NDVI"] - row["Harvest_NDVI"]) <= rule.get("min_drop", 0.20):
        reasons.append("drop")
    if (row["Peak_NDVI"] - row["Sowing_NDVI"]) <= rule.get("min_rise", 0.15):
        reasons.append("rise")

    return len(reasons) == 0


def _build_feature_table(raw: pd.DataFrame) -> pd.DataFrame:
    lifecycle_columns = [
        "Sowing_Date",
        "Peak_Date",
        "Harvest_Date",
    ]
    records = []

    for crop in raw["Crop"].unique():
        cfg = CROP_CONFIG.get(crop, DEFAULT_CFG)
        crop_df = raw[raw["Crop"] == crop]

        for farm, grp in crop_df.groupby("Farm"):
            grp = grp.reset_index(drop=True)
            dates = pd.to_datetime(grp["Date"].values).values.astype("datetime64[ns]")
            ndvi = grp["NDVI"].values
            ground_truth_date = pd.Timestamp(grp["Ground_Truth_Date"].iloc[0])

            segments = split_into_years(dates)
            smoothed_full = np.full(len(dates), np.nan, dtype=float)
            all_cycles = []

            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                mask = (dates >= seg_start) & (dates < seg_end)
                seg_idx_arr = np.where(mask)[0]
                if len(seg_idx_arr) < 5:
                    continue

                cycles, smoothed_seg = detect_cycles(dates[seg_idx_arr], ndvi[seg_idx_arr], cfg)
                smoothed_full[seg_idx_arr] = smoothed_seg

                for cycle in cycles:
                    all_cycles.append(
                        {
                            "sowing": int(seg_idx_arr[cycle["sowing"]]),
                            "peak": int(seg_idx_arr[cycle["peak"]]),
                            "harvest": int(seg_idx_arr[cycle["harvest"]]),
                            "seg_idx": seg_idx,
                        }
                    )

            containing_cycles = [
                cycle
                for cycle in all_cycles
                if pd.Timestamp(dates[cycle["sowing"]])
                <= ground_truth_date
                <= pd.Timestamp(dates[cycle["harvest"]])
            ]
            if containing_cycles:
                all_cycles = [
                    min(
                        containing_cycles,
                        key=lambda cycle: abs(
                            (
                                pd.Timestamp(dates[cycle["peak"]])
                                - ground_truth_date
                            ).days
                        ),
                    )
                ]
            else:
                all_cycles = []

            for cycle in sorted(all_cycles, key=lambda x: x["peak"]):
                sow_idx, peak_idx, harv_idx = cycle["sowing"], cycle["peak"], cycle["harvest"]
                sow_date = pd.Timestamp(dates[sow_idx])
                peak_date = pd.Timestamp(dates[peak_idx])
                harv_date = pd.Timestamp(dates[harv_idx])

                mask = (dates >= sow_date) & (dates <= harv_date)
                seg_days = ((dates[mask] - np.datetime64(sow_date)) / np.timedelta64(1, "D")).astype(float)
                seg_ndvi = ndvi[mask]

                if len(seg_days) > 1:
                    auc = float(_trapz(seg_ndvi, seg_days))
                else:
                    auc = np.nan

                rise_days = max((peak_date - sow_date).days, 1)
                fall_days = max((harv_date - peak_date).days, 1)

                row = {
                    "Crop": crop,
                    "Farm": farm,
                    "Sowing_Date": sow_date,
                    "Sowing_Month": sow_date.month,
                    "Peak_Date": peak_date,
                    "Peak_Month": peak_date.month,
                    "Harvest_Date": harv_date,
                    "Harvest_Month": harv_date.month,
                    "Duration_Days": int((harv_date - sow_date).days),
                    "Peak_NDVI": round(float(smoothed_full[peak_idx]), 4),
                    "Sowing_NDVI": round(float(smoothed_full[sow_idx]), 4),
                    "Harvest_NDVI": round(float(smoothed_full[harv_idx]), 4),
                    "AUC": round(auc, 2) if not np.isnan(auc) else np.nan,
                    "Mean_NDVI": round(float(np.nanmean(seg_ndvi)), 4),
                    "Std_NDVI": round(float(np.std(seg_ndvi)), 4),
                    "NDVI_range": round(float(np.ptp(seg_ndvi)), 4),
                    "Rise_rate": round((smoothed_full[peak_idx] - smoothed_full[sow_idx]) / rise_days, 5),
                    "Fall_rate": round((smoothed_full[peak_idx] - smoothed_full[harv_idx]) / fall_days, 5),
                }

                if not _validate_cycle(crop, pd.Series(row)):
                    continue

                records.append({**row})

    if not records:
        return pd.DataFrame(columns=["Crop", "Farm"] + lifecycle_columns + FEATURE_COLS)

    features_df = pd.DataFrame(records)
    return features_df[
        ["Crop", "Farm"] + lifecycle_columns + FEATURE_COLS
    ].dropna(subset=["Crop"] + FEATURE_COLS).reset_index(drop=True)


def _save_misclassified_plots(
    source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    misclassified: pd.DataFrame,
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_plot in output_dir.glob("*_misclassified.png"):
        old_plot.unlink()

    saved_count = 0

    source_df = source_df.copy()
    source_df["Date"] = pd.to_datetime(source_df["Date"], errors="coerce")
    source_df["NDVI"] = pd.to_numeric(source_df["NDVI"], errors="coerce")
    source_df = source_df.dropna(subset=["Farm_ID", "Date", "NDVI"])

    for farm_id in sorted(misclassified["Farm_ID"].astype(str).unique()):
        farm_rows = source_df[source_df["Farm_ID"].astype(str) == farm_id].sort_values("Date")
        feature_rows = feature_df[feature_df["Farm"].astype(str) == farm_id]
        prediction_rows = misclassified[misclassified["Farm_ID"].astype(str) == farm_id]
        if farm_rows.empty or feature_rows.empty or prediction_rows.empty:
            continue

        truth = str(prediction_rows.iloc[0]["Ground_Truth"])
        predicted = ", ".join(sorted(prediction_rows["Predicted"].astype(str).unique()))
        ground_truth_dates = pd.to_datetime(
            farm_rows["ground truth month"], errors="coerce"
        ).dropna().unique()

        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.plot(
            farm_rows["Date"],
            farm_rows["NDVI"],
            color="gray",
            linewidth=1.2,
            marker="o",
            markersize=2.5,
            alpha=0.8,
            label="NDVI",
        )

        for cycle_number, (_, cycle) in enumerate(feature_rows.iterrows(), start=1):
            ax.scatter(cycle["Sowing_Date"], cycle["Sowing_NDVI"], color="green", s=65, zorder=5)
            ax.scatter(cycle["Peak_Date"], cycle["Peak_NDVI"], color="red", s=65, zorder=5)
            ax.scatter(cycle["Harvest_Date"], cycle["Harvest_NDVI"], color="blue", s=65, zorder=5)
            ax.axvspan(cycle["Sowing_Date"], cycle["Harvest_Date"], color="steelblue", alpha=0.08)
            ax.text(
                cycle["Peak_Date"],
                cycle["Peak_NDVI"],
                f"  cycle {cycle_number}",
                fontsize=8,
                va="bottom",
            )

        for ground_truth_date in ground_truth_dates:
            ax.axvline(
                pd.Timestamp(ground_truth_date),
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="Ground Truth Date",
            )

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="best", fontsize=8)
        ax.set_title(
            f"{farm_id} | Actual: {truth} | Predicted: {predicted}\n"
            "Misclassified unseen lifecycle",
            fontsize=11,
        )
        ax.set_ylabel("NDVI")
        ax.set_ylim(bottom=min(0, float(farm_rows["NDVI"].min()) - 0.05))
        ax.grid(True, alpha=0.35)
        ax.margins(x=0.03)
        ax.tick_params(axis="x", rotation=45)
        fig.tight_layout()

        plot_path = output_dir / f"{farm_id}_misclassified.png"
        fig.savefig(plot_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved_count += 1

    return saved_count


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "3 to test.csv"
    model_path = base_dir / "model" / "crop_classifier.pkl"

    print(f"Loading unseen data from {csv_path}...")
    raw = _prepare_unseen_data(csv_path)
    feature_df = _build_feature_table(raw)

    if feature_df.empty:
        print("No valid feature rows could be generated for the unseen dataset.")
        return

    print(f"Generated {len(feature_df)} valid evaluation samples")

    bundle = joblib.load(model_path)
    model = bundle["model"]
    feature_cols = bundle.get("feature_cols", FEATURE_COLS)

    X = feature_df[feature_cols].values
    y_true = feature_df["Crop"].values

    y_pred = model.predict(X)
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

    scores = []
    for idx, (truth, pred) in enumerate(zip(y_true, y_pred)):
        conf = 0.0
        if proba is not None:
            class_index = np.where(model.classes_ == pred)[0]
            if len(class_index) > 0:
                conf = float(proba[idx, class_index[0]])
        scores.append({
            "Farm_ID": str(feature_df.iloc[idx]["Farm"]),
            "Ground_Truth": truth,
            "Predicted": pred,
            "Correct": bool(truth == pred),
            "Confidence": round(conf, 6),
        })

    predictions_df = pd.DataFrame(scores)
    predictions_df.to_csv(base_dir / "unseen_predictions.csv", index=False)

    print("\nNumber of samples evaluated:", len(y_true))
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Precision:", round(precision_score(y_true, y_pred, average="weighted", zero_division=0), 4))
    print("Recall:", round(recall_score(y_true, y_pred, average="weighted", zero_division=0), 4))
    print("F1-score:", round(f1_score(y_true, y_pred, average="weighted", zero_division=0), 4))
    print("\nClassification Report")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("\nConfusion Matrix")
    print(confusion_matrix(y_true, y_pred))

    misclassified = predictions_df[~predictions_df["Correct"]]
    print("\nMisclassified crop IDs")
    if misclassified.empty:
        print("None")
    else:
        print(
            misclassified[
                ["Farm_ID", "Ground_Truth", "Predicted"]
            ].to_string(index=False)
        )

    misclassified_ids = set(misclassified["Farm_ID"].astype(str))
    source_columns = [
        "crop label",
        "Farm_ID",
        "NDVI",
        "Date",
        "ground truth month",
        "LATITUDE",
        "LONGITUDE",
        "Lat, Long",
    ]
    source_df = pd.read_csv(csv_path)
    missing_columns = [column for column in source_columns if column not in source_df.columns]
    if missing_columns:
        print(f"\nCould not export misclassified source rows; missing columns: {missing_columns}")
    else:
        misclassified_source = source_df[
            source_df["Farm_ID"].astype(str).isin(misclassified_ids)
        ][source_columns]
        misclassified_path = base_dir / "misclassified_unseen_data.csv"
        misclassified_source.to_csv(misclassified_path, index=False)
        print(f"\nSaved all source rows for misclassified IDs to {misclassified_path}")

        plot_dir = base_dir / "lifecycle_plots" / "test_unseen_missclassified"
        plot_count = _save_misclassified_plots(
            source_df,
            feature_df,
            misclassified,
            plot_dir,
        )
        print(f"Saved {plot_count} misclassified lifecycle plots to {plot_dir}")

    print(f"\nSaved predictions to {base_dir / 'unseen_predictions.csv'}")


if __name__ == "__main__":
    main()
