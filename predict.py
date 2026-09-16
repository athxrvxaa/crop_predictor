"""
predict.py
----------
Loads trained model and runs prediction for a given lat/lon/date.
Called by app.py.
"""

import numpy as np
import pandas as pd
import joblib

# predict.py — update imports
from pipeline import (
    CROP_CONFIG,
    GENERIC_CFG,
    FEATURE_COLS,
    detect_cycles,
    build_features_from_cycle,
    find_valid_cycle_for_crop,
    validate_crop,
)
from gee_fetch import get_ndvi_timeseries

_model_bundle = None


def load_model(path: str = 'model/crop_classifier.pkl'):
    global _model_bundle
    if _model_bundle is None:
        _model_bundle = joblib.load(path)
    return _model_bundle['model'], _model_bundle['feature_cols']


def predict_crop(lat: float, lon: float, ref_date: str,
                 window_months: int = 18,
                 ground_truth: str = None,
                 gee_project: str = 'gee-project-497010'):
    clf, feature_cols = load_model()
    ref_date   = pd.Timestamp(ref_date)
    start_date = (ref_date - pd.DateOffset(months=window_months)).strftime('%Y-%m-%d')
    end_date   = (ref_date + pd.DateOffset(months=window_months)).strftime('%Y-%m-%d')

    ndvi_df = get_ndvi_timeseries(lat, lon, start_date, end_date, project=gee_project)
    if len(ndvi_df) < 5:
        return {'error': 'Not enough clean NDVI observations in this window'}

    dates  = pd.to_datetime(ndvi_df['Date'].values).values.astype('datetime64[ns]')
    ndvi   = ndvi_df['NDVI'].values
    fallback_smoothed = detect_cycles(dates, ndvi, GENERIC_CFG)[1]

    valid_candidates = []
    for crop in clf.classes_:
        crop_cfg = CROP_CONFIG.get(crop, GENERIC_CFG)
        candidate_cycles, crop_smoothed = detect_cycles(dates, ndvi, crop_cfg)
        if not candidate_cycles:
            continue

        cycle = find_valid_cycle_for_crop(
            crop,
            dates,
            ndvi,
            crop_cfg,
            ref_date=ref_date,
            candidate_cycles=candidate_cycles,
        )
        if cycle is None:
            continue

        result = build_features_from_cycle(dates, ndvi, crop_smoothed, cycle)
        if result is None:
            continue

        feats, sow_date, peak_date, harv_date = result
        cycle_segment = ndvi[cycle['sowing']:cycle['harvest'] + 1]
        if not validate_crop(crop, sow_date, peak_date, harv_date, cycle_segment):
            continue

        valid_candidates.append((crop, cycle, feats, sow_date, peak_date, harv_date, crop_smoothed, crop_cfg['smooth']))

    if not valid_candidates:
        return {
            'lat': lat,
            'lon': lon,
            'ref_date': str(ref_date.date()),
            'predicted_crop': 'No confident crop match',
            'confidence_pct': 0.0,
            'top3': [],
            'sowing_date': None,
            'peak_date': None,
            'harvest_date': None,
            'ndvi_dates': [str(d.date()) for d in ndvi_df['Date']],
            'ndvi_values': [round(float(v), 4) for v in ndvi],
            'smoothed': [round(float(v), 4) if not np.isnan(v) else None for v in fallback_smoothed],
            'smooth_window': GENERIC_CFG['smooth'],
            'ref_date_inside_cycle': False,
        }

    crop_probabilities = {}
    for crop, cycle, feats, sow_date, peak_date, harv_date, crop_smoothed, smooth_window in valid_candidates:
        X = pd.DataFrame([feats])[feature_cols].values
        probs = clf.predict_proba(X)[0]
        crop_idx = int(np.where(clf.classes_ == crop)[0][0])
        crop_probabilities[crop] = float(probs[crop_idx])

    total_prob = float(sum(crop_probabilities.values()))
    if total_prob <= 0:
        return {
            'lat': lat,
            'lon': lon,
            'ref_date': str(ref_date.date()),
            'predicted_crop': 'No confident crop match',
            'confidence_pct': 0.0,
            'top3': [],
            'sowing_date': None,
            'peak_date': None,
            'harvest_date': None,
            'ndvi_dates': [str(d.date()) for d in ndvi_df['Date']],
            'ndvi_values': [round(float(v), 4) for v in ndvi],
            'smoothed': [round(float(v), 4) if not np.isnan(v) else None for v in fallback_smoothed],
            'smooth_window': GENERIC_CFG['smooth'],
            'ref_date_inside_cycle': False,
        }

    normalized_probs = {crop: prob / total_prob for crop, prob in crop_probabilities.items()}
    ranked_crops = sorted(normalized_probs.items(), key=lambda item: item[1], reverse=True)
    top_crop, top_prob = ranked_crops[0]

    top_entry = next(entry for entry in valid_candidates if entry[0] == top_crop)
    top_cycle = top_entry[1]
    _, _, feats, sow_date, peak_date, harv_date, crop_smoothed, smooth_window = top_entry

    output = {
        'lat': lat,
        'lon': lon,
        'ref_date': str(ref_date.date()),
        'predicted_crop':   top_crop,
        'confidence_pct':   round(top_prob * 100, 2),
        'top3': [
            {'crop': crop, 'confidence': round(prob * 100, 2)}
            for crop, prob in ranked_crops[:3]
        ],
        'sowing_date':   str(pd.Timestamp(sow_date).date()),
        'peak_date':     str(pd.Timestamp(peak_date).date()),
        'harvest_date':  str(pd.Timestamp(harv_date).date()),
        'sowing_idx':    int(top_cycle['sowing']),
        'peak_idx':      int(top_cycle['peak']),
        'harvest_idx':   int(top_cycle['harvest']),
        'ndvi_dates':  [str(d.date()) for d in ndvi_df['Date']],
        'ndvi_values': [round(float(v), 4) for v in ndvi],
        'smoothed':    [round(float(v), 4) if not np.isnan(v) else None for v in crop_smoothed],
        'smooth_window': smooth_window,
        'ref_date_inside_cycle': bool(
            pd.Timestamp(sow_date) <= ref_date <= pd.Timestamp(harv_date)
        ),
    }

    if ground_truth:
        output['ground_truth'] = ground_truth
        output['correct']      = output['predicted_crop'] == ground_truth

    return output
