import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from pipeline import (
    CROP_CONFIG,
    DEFAULT_CFG,
    RULES,
    _trapz,
    detect_cycles,
    split_into_years,
)


parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="all_5_crops_combined.csv", help="Input NDVI CSV")
parser.add_argument("--outdir", default="lifecycle_plots", help="Folder to write plots into")
args = parser.parse_args()


raw = pd.read_csv(args.csv)


def _norm(col):
    return col.strip().lower().replace(" ", "_")


rename_map = {}
for c in raw.columns:
    n = _norm(c)
    if n in ("farm", "farm_id", "farmid"):
        rename_map[c] = "Farm"
    elif n in (
        "crop",
        "crop_label",
        "crop_name",
        "cropname",
        "crop_type",
        "croptype",
        "label",
        "class",
    ):
        rename_map[c] = "Crop"
    elif n in ("date", "sentinel_date", "image_date"):
        rename_map[c] = "Date"
    elif n == "ndvi":
        rename_map[c] = "NDVI"

raw = raw.rename(columns=rename_map)
raw["Date"] = pd.to_datetime(raw["Date"], format="%d-%m-%Y", errors="coerce")
raw["NDVI"] = pd.to_numeric(raw["NDVI"], errors="coerce")
raw = raw.dropna(subset=["Date", "NDVI", "Farm", "Crop"])
raw = raw[raw["NDVI"] >= 0]

if raw["Farm"].nunique() <= 1 and "Latitude" in raw.columns:
    coords = raw[["Latitude", "Longitude"]].drop_duplicates().reset_index(drop=True)
    coords["Farm"] = [f"F{i + 1:02d}" for i in range(len(coords))]
    raw = raw.drop(columns=["Farm"]).merge(coords, on=["Latitude", "Longitude"], how="left")

if "Latitude" in raw.columns:
    raw = raw.groupby(["Farm", "Crop", "Date"], as_index=False)["NDVI"].mean()

ndvi_df = raw.sort_values(["Crop", "Farm", "Date"]).reset_index(drop=True)

results_by_farm = {}
for crop in ndvi_df["Crop"].unique():
    cfg = CROP_CONFIG.get(crop, DEFAULT_CFG)
    for farm, grp in ndvi_df[ndvi_df["Crop"] == crop].groupby("Farm"):
        grp = grp.reset_index(drop=True)
        dates = pd.to_datetime(grp["Date"].values).values.astype("datetime64[ns]")
        ndvi = grp["NDVI"].values
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
            for c in cycles:
                all_cycles.append({
                    "sowing": seg_idx_arr[c["sowing"]],
                    "peak": seg_idx_arr[c["peak"]],
                    "harvest": seg_idx_arr[c["harvest"]],
                    "seg_idx": seg_idx,
                })

        results_by_farm[(crop, farm)] = {
            "dates": dates,
            "ndvi": ndvi,
            "smoothed": smoothed_full,
            "cycles": all_cycles,
        }

records = []
for (crop, farm), result in results_by_farm.items():
    dates = result["dates"]
    ndvi = result["ndvi"]
    smoothed = result["smoothed"]
    seg_counters = {}

    for cycle in sorted(result["cycles"], key=lambda x: x["peak"]):
        seg_idx = cycle["seg_idx"]
        seg_counters[seg_idx] = seg_counters.get(seg_idx, 0) + 1
        cycle_num = seg_counters[seg_idx]

        sow_date = pd.Timestamp(dates[cycle["sowing"]])
        peak_date = pd.Timestamp(dates[cycle["peak"]])
        harv_date = pd.Timestamp(dates[cycle["harvest"]])

        mask = (dates >= sow_date) & (dates <= harv_date)
        seg_days = ((dates[mask] - np.datetime64(sow_date)) / np.timedelta64(1, "D")).astype(float)
        seg_ndvi = ndvi[mask]
        auc = float(_trapz(seg_ndvi, seg_days)) if len(seg_days) > 1 else np.nan

        records.append({
            "Crop": crop,
            "Farm": farm,
            "Year_Segment": seg_idx + 1,
            "Cycle": cycle_num,
            "Sowing_Date": sow_date,
            "Sowing_Month": sow_date.month,
            "Sowing_NDVI": round(float(smoothed[cycle["sowing"]]), 4),
            "Peak_Date": peak_date,
            "Peak_Month": peak_date.month,
            "Peak_NDVI": round(float(smoothed[cycle["peak"]]), 4),
            "Harvest_Date": harv_date,
            "Harvest_Month": harv_date.month,
            "Harvest_NDVI": round(float(smoothed[cycle["harvest"]]), 4),
            "Duration_Days": (harv_date - sow_date).days,
            "AUC": round(auc, 2) if not pd.isna(auc) else np.nan,
            "Mean_NDVI": round(float(np.nanmean(seg_ndvi)), 4),
        })

lifecycle_df = pd.DataFrame(records)


def validate(row):
    rule = RULES.get(row["Crop"])
    if rule is None:
        return "No rule defined"

    reasons = []
    if row["Sowing_Month"] not in rule["sow_months"]:
        reasons.append(f"Sow month {row['Sowing_Month']} invalid")
    if row["Harvest_Month"] not in rule["harv_months"]:
        reasons.append(f"Harvest month {row['Harvest_Month']} invalid")
    if not (rule["min_dur"] <= row["Duration_Days"] <= rule["max_dur"]):
        reasons.append(f"Duration {row['Duration_Days']}d not in [{rule['min_dur']}-{rule['max_dur']}]")
    if row["Peak_NDVI"] - row["Harvest_NDVI"] <= rule.get("min_drop", 0.20):
        reasons.append("NDVI drop too small")
    if row["Peak_NDVI"] - row["Sowing_NDVI"] <= rule.get("min_rise", 0.15):
        reasons.append("NDVI rise too small")
    return "Valid" if len(reasons) == 0 else "; ".join(reasons)

lifecycle_df["Validation"] = lifecycle_df.apply(validate, axis=1)
lifecycle_df["Is_Valid"] = lifecycle_df["Validation"] == "Valid"
valid_df = lifecycle_df[lifecycle_df["Is_Valid"]].reset_index(drop=True)
invalid_df = lifecycle_df[~lifecycle_df["Is_Valid"]].reset_index(drop=True)

valid_dir = os.path.join(args.outdir, "valid")
invalid_dir = os.path.join(args.outdir, "invalid")
os.makedirs(valid_dir, exist_ok=True)
os.makedirs(invalid_dir, exist_ok=True)

handles = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=9, label='Sowing'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=9, label='Peak'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=9, label='Harvest'),
]


def save_plot(row, folder, title_suffix):
    farm_data = ndvi_df[
        (ndvi_df["Crop"] == row["Crop"])
        & (ndvi_df["Farm"] == row["Farm"])
        & (ndvi_df["Date"] >= row["Sowing_Date"])
        & (ndvi_df["Date"] <= row["Harvest_Date"])
    ].sort_values("Date")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(farm_data["Date"], farm_data["NDVI"], color="gray", linewidth=1.2, marker="o", markersize=2.5, alpha=0.8)
    ax.scatter(row["Sowing_Date"], row["Sowing_NDVI"], color="green", s=80, zorder=5)
    ax.scatter(row["Peak_Date"], row["Peak_NDVI"], color="red", s=80, zorder=5)
    ax.scatter(row["Harvest_Date"], row["Harvest_NDVI"], color="blue", s=80, zorder=5)
    ax.set_title(f"{row['Crop']} | {row['Farm']} | {title_suffix}\nDuration={row['Duration_Days']}d", fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_ylabel("NDVI")
    ax.grid(True, alpha=0.35)
    ax.margins(x=0.06)
    ax.tick_params(axis='x', rotation=45)
    ax.legend(handles=handles, loc='upper right', fontsize=8)
    fig.tight_layout()

    fname = f"{row['Crop']}_{row['Farm']}_cycle{row['Cycle']}_{row['Year_Segment']}.png"
    fig.savefig(os.path.join(folder, fname), dpi=180, bbox_inches="tight")
    plt.close(fig)


for _, row in valid_df.iterrows():
    save_plot(row, valid_dir, "VALID")

for _, row in invalid_df.iterrows():
    save_plot(row, invalid_dir, "INVALID")

print(f"Saved valid lifecycle plots to: {valid_dir}")
print(f"Saved invalid lifecycle plots to: {invalid_dir}")
print(f"Valid rows: {len(valid_df)} | Invalid rows: {len(invalid_df)}")
