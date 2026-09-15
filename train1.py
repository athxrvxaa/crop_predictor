"""
train.py

Run once to train the Random Forest classifier and save it as
model/crop_classifier.pkl

Usage
-----
python train.py --csv all_5_crops_combined.csv
"""

import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from pipeline import (
    CROP_CONFIG,
    DEFAULT_CFG,
    FEATURE_COLS,
    RULES,
    _trapz,
    detect_cycles,
    build_features_from_cycle,
)

# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument(
    "--csv",
    default="all_5_crops_combined.csv",
    help="Input NDVI CSV",
)
args = parser.parse_args()

# ------------------------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------------------------

print(f"Loading {args.csv} ...")

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

    elif n in (
        "date",
        "sentinel_date",
        "image_date",
    ):
        rename_map[c] = "Date"

    elif n == "ndvi":
        rename_map[c] = "NDVI"


raw = raw.rename(columns=rename_map)

print("Columns after renaming:", raw.columns.tolist())

raw["Date"] = pd.to_datetime(
    raw["Date"],
    format="%d-%m-%Y",
    errors="coerce",
)
raw["NDVI"] = pd.to_numeric(raw["NDVI"], errors="coerce")

raw = raw.dropna(
    subset=[
        "Date",
        "NDVI",
        "Farm",
        "Crop",
    ]
)

raw = raw[raw["NDVI"] >= 0]

# ------------------------------------------------------------------------------
# If farm ids are missing but lat/lon exist
# ------------------------------------------------------------------------------

if raw["Farm"].nunique() <= 1 and "Latitude" in raw.columns:

    coords = (
        raw[["Latitude", "Longitude"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    coords["Farm"] = [
        f"F{i+1:02d}"
        for i in range(len(coords))
    ]

    raw = (
        raw.drop(columns=["Farm"])
        .merge(
            coords,
            on=["Latitude", "Longitude"],
            how="left",
        )
    )

# ------------------------------------------------------------------------------
# Remove duplicate dates
# ------------------------------------------------------------------------------

if "Latitude" in raw.columns:

    raw = (
        raw.groupby(
            ["Farm", "Crop", "Date"],
            as_index=False,
        )["NDVI"]
        .mean()
    )

ndvi_df = (
    raw.sort_values(
        ["Crop", "Farm", "Date"]
    )
    .reset_index(drop=True)
)

print(
    f"Loaded {len(ndvi_df)} rows | Crops: "
    f"{sorted(ndvi_df['Crop'].unique())}"
)

# ------------------------------------------------------------------------------
# DETECT LIFECYCLES
# ------------------------------------------------------------------------------

print("\nDetecting lifecycles ...")

results_by_farm = {}

for crop in ndvi_df["Crop"].unique():

    cfg = CROP_CONFIG.get(crop, DEFAULT_CFG)

    crop_df = ndvi_df[
        ndvi_df["Crop"] == crop
    ]

    for farm, grp in crop_df.groupby("Farm"):

        grp = grp.reset_index(drop=True)

        dates = (
            pd.to_datetime(grp["Date"].values)
            .values
            .astype("datetime64[ns]")
        )

        ndvi = grp["NDVI"].values

        cycles, smoothed = detect_cycles(
            dates,
            ndvi,
            cfg,
        )

        results_by_farm[(crop, farm)] = {
            "dates": dates,
            "ndvi": ndvi,
            "smoothed": smoothed,
            "cycles": cycles,
        }

print(f"Processed {len(results_by_farm)} (crop, farm) combinations")

# ------------------------------------------------------------------------------
# BUILD LIFECYCLE TABLE
# ------------------------------------------------------------------------------

records = []

for (crop, farm), result in results_by_farm.items():

    dates = result["dates"]
    ndvi = result["ndvi"]
    smoothed = result["smoothed"]

    for cycle in sorted(
        result["cycles"],
        key=lambda x: x["peak"],
    ):

        sow_date = pd.Timestamp(
            dates[cycle["sowing"]]
        )

        peak_date = pd.Timestamp(
            dates[cycle["peak"]]
        )

        harv_date = pd.Timestamp(
            dates[cycle["harvest"]]
        )

        mask = (
            (dates >= sow_date)
            &
            (dates <= harv_date)
        )

        seg_days = (
            (dates[mask] - np.datetime64(sow_date))
            / np.timedelta64(1, "D")
        ).astype(float)

        seg_ndvi = ndvi[mask]

        if len(seg_days) > 1:
            auc = float(
                _trapz(
                    seg_ndvi,
                    seg_days,
                )
            )
        else:
            auc = np.nan

        records.append(
            {
                "Crop": crop,
                "Farm": farm,

                "Sowing_Date": sow_date,
                "Sowing_Month": sow_date.month,
                "Sowing_NDVI": round(
                    float(smoothed[cycle["sowing"]]),
                    4,
                ),

                "Peak_Date": peak_date,
                "Peak_Month": peak_date.month,
                "Peak_NDVI": round(
                    float(smoothed[cycle["peak"]]),
                    4,
                ),

                "Harvest_Date": harv_date,
                "Harvest_Month": harv_date.month,
                "Harvest_NDVI": round(
                    float(smoothed[cycle["harvest"]]),
                    4,
                ),

                "Duration_Days": (
                    harv_date - sow_date
                ).days,

                "AUC": (
                    round(auc, 2)
                    if not np.isnan(auc)
                    else np.nan
                ),

                "Mean_NDVI": round(
                    float(np.nanmean(seg_ndvi)),
                    4,
                ),
            }
        )

lifecycle_df = pd.DataFrame(records)

print(f"Total cycles: {len(lifecycle_df)}")
# ------------------------------------------------------------------------------
# VALIDATE DETECTED LIFECYCLES
# ------------------------------------------------------------------------------

def validate(row):

    rule = RULES.get(row["Crop"])

    if rule is None:
        return "No rule defined"

    reasons = []

    if row["Sowing_Month"] not in rule["sow_months"]:
        reasons.append(
            f"Sow month {row['Sowing_Month']} invalid"
        )

    if row["Harvest_Month"] not in rule["harv_months"]:
        reasons.append(
            f"Harvest month {row['Harvest_Month']} invalid"
        )

    if not (
        rule["min_dur"]
        <= row["Duration_Days"]
        <= rule["max_dur"]
    ):
        reasons.append(
            f"Duration {row['Duration_Days']}d not in "
            f"[{rule['min_dur']}-{rule['max_dur']}]"
        )

    if (
        row["Peak_NDVI"] - row["Harvest_NDVI"]
        <= rule.get("min_drop", 0.20)
    ):
        reasons.append("NDVI drop too small")

    if (
        row["Peak_NDVI"] - row["Sowing_NDVI"]
        <= rule.get("min_rise", 0.15)
    ):
        reasons.append("NDVI rise too small")

    return "Valid" if len(reasons) == 0 else "; ".join(reasons)


lifecycle_df["Validation"] = lifecycle_df.apply(
    validate,
    axis=1,
)

lifecycle_df["Is_Valid"] = (
    lifecycle_df["Validation"] == "Valid"
)

valid_df = (
    lifecycle_df[lifecycle_df["Is_Valid"]]
    .reset_index(drop=True)
)

print(
    f"Valid: {len(valid_df)} | "
    f"Invalid: {len(lifecycle_df)-len(valid_df)}"
)

print(
    valid_df.groupby("Crop")
    .size()
    .rename("valid_cycles")
)

# ------------------------------------------------------------------------------
# FEATURE EXTRACTION
# ------------------------------------------------------------------------------

feature_records = []

for _, row in valid_df.iterrows():

    farm_data = (
        ndvi_df[
            (ndvi_df["Crop"] == row["Crop"])
            &
            (ndvi_df["Farm"] == row["Farm"])
            &
            (ndvi_df["Date"] >= row["Sowing_Date"])
            &
            (ndvi_df["Date"] <= row["Harvest_Date"])
        ]
        .sort_values("Date")
    )

    ndvi_vals = farm_data["NDVI"].values

    if len(ndvi_vals) < 3:
        continue

    rise_days = max(
        (row["Peak_Date"] - row["Sowing_Date"]).days,
        1,
    )

    fall_days = max(
        (row["Harvest_Date"] - row["Peak_Date"]).days,
        1,
    )

    feature_records.append({

        "Crop": row["Crop"],
        "Farm": row["Farm"],

        "Duration_Days": row["Duration_Days"],

        "Peak_NDVI": row["Peak_NDVI"],
        "Peak_Month": row["Peak_Month"],

        "Sowing_NDVI": row["Sowing_NDVI"],
        "Sowing_Month": row["Sowing_Month"],

        "Harvest_NDVI": row["Harvest_NDVI"],
        "Harvest_Month": row["Harvest_Month"],

        "AUC": row["AUC"],
        "Mean_NDVI": row["Mean_NDVI"],

        "Std_NDVI": round(
            float(np.std(ndvi_vals)),
            4,
        ),

        "NDVI_range": round(
            float(np.ptp(ndvi_vals)),
            4,
        ),

        "Rise_rate": round(
            (
                row["Peak_NDVI"]
                - row["Sowing_NDVI"]
            )
            / rise_days,
            5,
        ),

        "Fall_rate": round(
            (
                row["Peak_NDVI"]
                - row["Harvest_NDVI"]
            )
            / fall_days,
            5,
        ),

    })

features_df = pd.DataFrame(feature_records)

print(f"\nFeature matrix: {features_df.shape}")

# ------------------------------------------------------------------------------
# TRAIN / VALIDATION / TEST SPLIT
# ------------------------------------------------------------------------------

ml_df = (
    features_df
    .dropna(subset=FEATURE_COLS + ["Crop"])
    .reset_index(drop=True)
)

class_counts = ml_df["Crop"].value_counts()
unsupported_crops = class_counts[class_counts < 2].index.tolist()

if unsupported_crops:
    print(
        "Dropping crops with fewer than 2 valid samples "
        f"for stratified split: {unsupported_crops}"
    )
    ml_df = (
        ml_df[
            ~ml_df["Crop"].isin(unsupported_crops)
        ]
        .reset_index(drop=True)
    )

farm_crop = (
    ml_df[
        ["Farm", "Crop"]
    ]
    .drop_duplicates()
    .groupby("Farm", as_index=False)
    .first()
)

farms_arr = farm_crop["Farm"].values
crops_arr = farm_crop["Crop"].values

sss1 = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42,
)

train_idx, test_idx = next(
    sss1.split(farms_arr, crops_arr)
)

farms_train = set(
    farms_arr[train_idx]
)

farms_test = set(
    farms_arr[test_idx]
)

ml_df["split"] = ml_df["Farm"].map(
    lambda farm:
        "train"
        if farm in farms_train
        else "test"
)

train_df = ml_df[
    ml_df["split"] == "train"
]

test_df = ml_df[
    ml_df["split"] == "test"
]
print("\n========== DATA SPLIT ==========")
print(f"Train samples : {len(train_df)}")
print(f"Test          : {len(test_df)}")

print("\nTrain crops")
print(train_df["Crop"].value_counts())

print("\nTest crops")
print(test_df["Crop"].value_counts())

# print(
#     f"\nTrain: {len(train_df)} | "
#     f"Val: {len(val_df)} | "
#     f"Test: {len(test_df)}"
# )

X_train = train_df[FEATURE_COLS].values
y_train = train_df["Crop"].values

X_test = test_df[FEATURE_COLS].values
y_test = test_df["Crop"].values

# ------------------------------------------------------------------------------
# TRAIN RANDOM FOREST
# ------------------------------------------------------------------------------

print("\nTraining Random Forest...")

clf = RandomForestClassifier(

    n_estimators=80,
    class_weight="balanced",
    random_state=42,

)

clf.fit(
    X_train,
    y_train,
)

print("\n========== TEST ==========")

if len(y_test) == 0:
    print("No test samples found.")
else:
    y_pred = clf.predict(X_test)

    print(classification_report(
        y_test,
        y_pred,
        zero_division=0
    ))

print("\n========== TRAIN ==========")

train_pred = clf.predict(X_train)

print(classification_report(
    y_train,
    train_pred,
    zero_division=0
))

if len(y_test) == 0:
    print("\nNo test samples available for confusion matrix")
else:
    cm = confusion_matrix(y_test, y_pred)

    print("\nConfusion Matrix")
    print(cm)

model_dir = os.path.join(os.path.dirname(__file__), "model")
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "crop_classifier.pkl")
joblib.dump(
    {
        "model": clf,
        "feature_cols": FEATURE_COLS,
    },
    model_path,
)
print(f"\nSaved trained model bundle to: {model_path}")


# if len(y_test):

#     print("\n----- TEST -----")

#     print(
#         classification_report(
#             y_test,
#             clf.predict(X_test),
#             zero_division=0,
#         )
#     )

# ------------------------------------------------------------------------------
# SAVE MODEL
# ------------------------------------------------------------------------------

joblib.dump(

    {
        "model": clf,
        "feature_cols": FEATURE_COLS,
    },

    "model/crop_classifier.pkl",

)

print(
    "\nModel saved to model/crop_classifier.pkl"
)


# """
# train.py
# --------
# Run this once to train the Random Forest from your CSV and save crop_classifier.pkl.

# Usage:
#     python train.py --csv all_5_crops_combined.csv

# The .pkl is then loaded by app.py for predictions.
# """

# import argparse
# import numpy as np
# import pandas as pd
# import joblib
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.metrics import classification_report, confusion_matrix

# from pipeline import (
#     CROP_CONFIG, DEFAULT_CFG, FEATURE_COLS, RULES, _trapz,
#     detect_cycles, build_features_from_cycle
# )

# # from pipeline import (
# #     CROP_CONFIG, DEFAULT_CFG, FEATURE_COLS, RULES, _trapz,
# #     split_into_years, detect_cycles, build_features_from_cycle
# # )

# # ── CLI ───────────────────────────────────────────────────────────────────────
# parser = argparse.ArgumentParser()
# parser.add_argument('--csv', default='all_5_crops_combined.csv', help='Path to input CSV')
# args = parser.parse_args()

# # ── Load & clean ──────────────────────────────────────────────────────────────
# print(f'Loading {args.csv} ...')
# raw = pd.read_csv(args.csv)

# def _norm(c):
#     return c.strip().lower().replace(' ', '_')
# rename_map = {}

# for c in raw.columns:
#     n = _norm(c)

#     if n in ('farm_id', 'farm', 'farmid'):
#         rename_map[c] = 'Farm'

#     elif n in (
#         'crop',
#         'crop_label',
#         'crop_name',
#         'cropname',
#         'crop_type',
#         'croptype',
#         'label',
#         'class'
#     ):
#         rename_map[c] = 'Crop'

#     elif n in ('sentinel_date', 'date', 'image_date'):
#         rename_map[c] = 'Date'

#     elif n == 'ndvi':
#         rename_map[c] = 'NDVI'

# raw = raw.rename(columns=rename_map)
# print("Columns after renaming:", raw.columns.tolist())
# raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')
# raw['NDVI'] = pd.to_numeric(raw['NDVI'], errors='coerce')
# raw = raw.dropna(subset=['Date', 'NDVI', 'Farm', 'Crop'])
# raw = raw[raw['NDVI'] >= 0]

# # Assign Farm_ID from lat/lon if Farm is all the same
# if raw['Farm'].nunique() <= 1 and 'Latitude' in raw.columns:
#     coords = raw[['Latitude','Longitude']].drop_duplicates().reset_index(drop=True)
#     coords['Farm'] = [f'F{i+1:02d}' for i in range(len(coords))]
#     raw = raw.drop(columns=['Farm']).merge(coords, on=['Latitude','Longitude'], how='left')

# # Dedup
# if 'Latitude' in raw.columns:
#     raw = raw.groupby(['Farm','Crop','Date'], as_index=False)['NDVI'].mean()

# ndvi_df = raw.sort_values(['Crop','Farm','Date']).reset_index(drop=True)
# print(f'Loaded {len(ndvi_df)} rows | Crops: {sorted(ndvi_df["Crop"].unique())}')

# # train.py — replace the imports line
# from pipeline import (
#     CROP_CONFIG, DEFAULT_CFG, FEATURE_COLS, RULES, _trapz,
#     detect_cycles, build_features_from_cycle
# )

# print('\nDetecting lifecycles ...')
# results_by_farm = {}
# # train.py — replace the lifecycle detection loop with this:
# for crop in ndvi_df['Crop'].unique():
#     cfg = CROP_CONFIG.get(crop, DEFAULT_CFG)
#     for farm, grp in ndvi_df[ndvi_df['Crop'] == crop].groupby('Farm'):
#         grp   = grp.reset_index(drop=True)
#         dates = pd.to_datetime(grp['Date'].values).values.astype('datetime64[ns]')
#         ndvi  = grp['NDVI'].values

#         cycles, smoothed = detect_cycles(dates, ndvi, cfg)

#         results_by_farm[(crop, farm)] = {
#             'dates': dates, 'ndvi': ndvi,
#             'smoothed': smoothed, 'cycles': cycles,
#         }

# # ── Lifecycle detection ───────────────────────────────────────────────────────
# # print('\nDetecting lifecycles ...')
# # results_by_farm = {}

# # for crop in ndvi_df['Crop'].unique():
# #     cfg = CROP_CONFIG.get(crop, DEFAULT_CFG)
# #     for farm, grp in ndvi_df[ndvi_df['Crop'] == crop].groupby('Farm'):
# #         grp   = grp.reset_index(drop=True)
# #         dates = pd.to_datetime(grp['Date'].values.astype('datetime64[ns]'))
# #         ndvi  = grp['NDVI'].values

# #         segments      = split_into_years(dates)
# #         smoothed_full = np.full(len(dates), np.nan)
# #         all_cycles    = []

# #         for seg_idx, (seg_start, seg_end) in enumerate(segments):
# #             mask        = (dates >= seg_start) & (dates < seg_end)
# #             seg_idx_arr = np.where(mask)[0]
# #             if len(seg_idx_arr) < 5:
# #                 continue
# #             cycles, smoothed_seg = detect_cycles(dates[seg_idx_arr], ndvi[seg_idx_arr], cfg)
# #             smoothed_full[seg_idx_arr] = smoothed_seg
# #             for c in cycles:
# #                 all_cycles.append({
# #                     'sowing':  seg_idx_arr[c['sowing']],
# #                     'peak':    seg_idx_arr[c['peak']],
# #                     'harvest': seg_idx_arr[c['harvest']],
# #                     'seg_idx': seg_idx,
# #                 })

# #         results_by_farm[(crop, farm)] = {
# #             'dates': dates, 'ndvi': ndvi, 'smoothed': smoothed_full,
# #             'cycles': all_cycles,
# #         }

# # print(f'Processed {len(results_by_farm)} (crop, farm) combinations')

# # ── Build lifecycle table ─────────────────────────────────────────────────────
# records = []
# for (crop, farm), r in results_by_farm.items():
#     dates, ndvi, smoothed = r['dates'], r['ndvi'], r['smoothed']
#     seg_counters = {}
#     for c in sorted(r['cycles'], key=lambda x: x['peak']):
#         seg_counters[c['seg_idx']] = seg_counters.get(c['seg_idx'], 0) + 1

#         sow_date  = pd.Timestamp(dates[c['sowing']])
#         peak_date = pd.Timestamp(dates[c['peak']])
#         harv_date = pd.Timestamp(dates[c['harvest']])

#         mask     = (dates >= sow_date) & (dates <= harv_date)
#         seg_days = pd.to_timedelta(dates[mask] - sow_date).days.astype(float)
#         seg_ndvi = ndvi[mask]
#         auc      = float(_trapz(seg_ndvi, seg_days)) if len(seg_days) > 1 else np.nan

#         records.append({
#             'Crop': crop, 'Farm': farm,
#             'Sowing_Date':   sow_date,  'Sowing_Month':  sow_date.month,
#             'Sowing_NDVI':   round(float(smoothed[c['sowing']]),   4),
#             'Peak_Date':     peak_date, 'Peak_Month':    peak_date.month,
#             'Peak_NDVI':     round(float(smoothed[c['peak']]),     4),
#             'Harvest_Date':  harv_date, 'Harvest_Month': harv_date.month,
#             'Harvest_NDVI':  round(float(smoothed[c['harvest']]),  4),
#             'Duration_Days': (harv_date - sow_date).days,
#             'AUC':           round(auc, 2) if not np.isnan(auc) else np.nan,
#             'Mean_NDVI':     round(float(np.nanmean(seg_ndvi)), 4),
#             'seg_idx':       c['seg_idx'],
#         })

# lifecycle_df = pd.DataFrame(records)
# print(f'Total cycles: {len(lifecycle_df)}')

# # ── Validation ────────────────────────────────────────────────────────────────
# def validate(row):
#     rule = RULES.get(row['Crop'])
#     if rule is None:
#         return 'No rule defined'
#     reasons = []
#     if row['Sowing_Month']  not in rule['sow_months']:
#         reasons.append(f"Sow month {row['Sowing_Month']} invalid")
#     if row['Harvest_Month'] not in rule['harv_months']:
#         reasons.append(f"Harvest month {row['Harvest_Month']} invalid")
#     if not (rule['min_dur'] <= row['Duration_Days'] <= rule['max_dur']):
#         reasons.append(f"Duration {row['Duration_Days']}d not in [{rule['min_dur']}-{rule['max_dur']}]")
#     if row['Peak_NDVI'] - row['Harvest_NDVI'] <= rule.get('min_drop', 0.2):
#         reasons.append('NDVI drop too small')
#     if row['Peak_NDVI'] - row['Sowing_NDVI']  <= rule.get('min_rise', 0.15):
#         reasons.append('NDVI rise too small')
#     return 'Valid' if not reasons else '; '.join(reasons)

# lifecycle_df['Validation'] = lifecycle_df.apply(validate, axis=1)
# lifecycle_df['Is_Valid']   = lifecycle_df['Validation'] == 'Valid'
# valid_df = lifecycle_df[lifecycle_df['Is_Valid']].reset_index(drop=True)
# print(f'Valid: {len(valid_df)}  |  Invalid: {len(lifecycle_df) - len(valid_df)}')
# print(valid_df.groupby('Crop').size().rename('valid_cycles'))

# # ── Feature extraction ────────────────────────────────────────────────────────
# feature_records = []
# for _, row in valid_df.iterrows():
#     farm_data = ndvi_df[
#         (ndvi_df['Crop'] == row['Crop']) & (ndvi_df['Farm'] == row['Farm']) &
#         (ndvi_df['Date'] >= row['Sowing_Date']) & (ndvi_df['Date'] <= row['Harvest_Date'])
#     ].sort_values('Date')

#     ndvi_vals = farm_data['NDVI'].values
#     if len(ndvi_vals) < 3:
#         continue

#     rise_days = max((row['Peak_Date'] - row['Sowing_Date']).days, 1)
#     fall_days = max((row['Harvest_Date'] - row['Peak_Date']).days, 1)

#     feature_records.append({
#         'Crop':          row['Crop'],
#         'Farm':          row['Farm'],
#         'Duration_Days': row['Duration_Days'],
#         'Peak_NDVI':     row['Peak_NDVI'],
#         'Peak_Month':    row['Peak_Month'],
#         'Sowing_NDVI':   row['Sowing_NDVI'],
#         'Sowing_Month':  row['Sowing_Month'],
#         'Harvest_NDVI':  row['Harvest_NDVI'],
#         'Harvest_Month': row['Harvest_Month'],
#         'AUC':           row['AUC'],
#         'Mean_NDVI':     row['Mean_NDVI'],
#         'Std_NDVI':      round(float(np.std(ndvi_vals)), 4),
#         'NDVI_range':    round(float(np.ptp(ndvi_vals)), 4),
#         'Rise_rate':     round((row['Peak_NDVI'] - row['Sowing_NDVI'])   / rise_days, 5),
#         'Fall_rate':     round((row['Peak_NDVI'] - row['Harvest_NDVI'])  / fall_days, 5),
#     })

# features_df = pd.DataFrame(feature_records)
# print(f'\nFeature matrix: {features_df.shape}')

# # ── Train / val / test split (farm-level, no leakage) ────────────────────────
# ml_df = features_df.dropna(subset=FEATURE_COLS + ['Crop']).reset_index(drop=True)
# farm_crop  = ml_df[['Farm','Crop']].drop_duplicates().reset_index(drop=True)
# farms_arr  = farm_crop['Farm'].values
# crops_arr  = farm_crop['Crop'].values

# sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
# train_val_idx, test_idx = next(sss1.split(farms_arr, crops_arr))

# sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.176, random_state=42)
# train_idx2, val_idx2 = next(sss2.split(farms_arr[train_val_idx], crops_arr[train_val_idx]))

# farms_train = set(farms_arr[train_val_idx][train_idx2])
# farms_val   = set(farms_arr[train_val_idx][val_idx2])
# farms_test  = set(farms_arr[test_idx])

# ml_df['split'] = ml_df['Farm'].apply(
#     lambda f: 'train' if f in farms_train else ('val' if f in farms_val else 'test')
# )

# train_df   = ml_df[ml_df['split'] == 'train']
# val_df_ml  = ml_df[ml_df['split'] == 'val']
# test_df_ml = ml_df[ml_df['split'] == 'test']

# print(f'\nTrain: {len(train_df)} | Val: {len(val_df_ml)} | Test: {len(test_df_ml)}')

# X_train = train_df[FEATURE_COLS].values
# y_train = train_df['Crop'].values
# X_val   = val_df_ml[FEATURE_COLS].values
# y_val   = val_df_ml['Crop'].values
# X_test  = test_df_ml[FEATURE_COLS].values
# y_test  = test_df_ml['Crop'].values

# # ── Train RF ──────────────────────────────────────────────────────────────────
# print('\nTraining Random Forest ...')
# clf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
# clf.fit(X_train, y_train)

# if len(y_val) > 0:
#     print('\n── Validation ──')
#     print(classification_report(y_val, clf.predict(X_val), zero_division=0))

# if len(y_test) > 0:
#     print('\n── Test ──')
#     print(classification_report(y_test, clf.predict(X_test), zero_division=0))

# # ── Save ──────────────────────────────────────────────────────────────────────
# joblib.dump({'model': clf, 'feature_cols': FEATURE_COLS}, 'model/crop_classifier.pkl')
# print('\nSaved → model/crop_classifier.pkl')
