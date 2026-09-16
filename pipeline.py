"""
pipeline.py
-----------
Core lifecycle detection + feature extraction + prediction logic.
Extracted from predct.ipynb. Used by both train.py and app.py.
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ── Crop configs ──────────────────────────────────────────────────────────────
CROP_CONFIG = {
    'Gram':        {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Lentil':      {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Maize':       {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Mustard':     {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Potato':      {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Rice':        {'smooth': 4, 'max_cycles': 1, 'search_months': 5, 'use_prominence': False, 'prominence': None},
    'Wheat':       {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Sugarcane':   {'smooth': 6, 'max_cycles': 1, 'search_months': 8, 'use_prominence': True,  'prominence': 0.08},
    'Banana':      {'smooth': 6, 'max_cycles': 1, 'search_months': 7, 'use_prominence': True,  'prominence': 0.06},
    'Grapes':      {'smooth': 3, 'max_cycles': 2, 'search_months': 3, 'use_prominence': True,  'prominence': 0.10},
    'Onion':       {'smooth': 2, 'max_cycles': 1, 'search_months': 3, 'use_prominence': False, 'prominence': None},
    'Tomato':      {'smooth': 3, 'max_cycles': 3, 'search_months': 2, 'use_prominence': False, 'prominence': None},
    'Paddy':       {'smooth': 3, 'max_cycles': 2, 'search_months': 5, 'use_prominence': True,  'prominence': 0.15},
    'Soybean':     {'smooth': 4, 'max_cycles': 1, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Cotton':      {'smooth': 2, 'max_cycles': 2, 'search_months': 6, 'use_prominence': True,  'prominence': 0.07},
    'Jowar':       {'smooth': 4, 'max_cycles': 2, 'search_months': 4, 'use_prominence': False, 'prominence': None},
    'Bajra':       {'smooth': 3, 'max_cycles': 1, 'search_months': 3, 'use_prominence': False, 'prominence': None},
    'Tur':         {'smooth': 5, 'max_cycles': 1, 'search_months': 6, 'use_prominence': False, 'prominence': None},
    'Turmeric':    {'smooth': 6, 'max_cycles': 1, 'search_months': 8, 'use_prominence': True,  'prominence': 0.06},
    'Pomegranate': {'smooth': 6, 'max_cycles': 2, 'search_months': 4, 'use_prominence': True,  'prominence': 0.07},
    'Safflower':   {'smooth': 3, 'max_cycles': 2, 'search_months': 3, 'use_prominence': True,  'prominence': 0.08},
    'Cabbage':     {'smooth': 3, 'max_cycles': 3, 'search_months': 2, 'use_prominence': False, 'prominence': None},
}

DEFAULT_CFG = {'smooth': 4, 'max_cycles': 1, 'search_months': 6, 'use_prominence': False, 'prominence': None}
GENERIC_CFG = {'smooth': 3, 'max_cycles': 1, 'search_months': 6, 'use_prominence': False, 'prominence': None}

FEATURE_COLS = [
    'Duration_Days', 'Peak_NDVI', 'Peak_Month',
    'Sowing_NDVI',   'Sowing_Month',
    'Harvest_NDVI',  'Harvest_Month',
    'AUC', 'Mean_NDVI', 'Std_NDVI', 'NDVI_range',
    'Rise_rate', 'Fall_rate',
]

RULES = {
    'Onion':     {'sow_months': [5,6,7,8,9,10,11,12], 'harv_months': [1,2,3,4,5,10,11,12], 'min_dur': 85,  'max_dur': 180, 'min_drop': 0.20, 'min_rise': 0.15},
    'Tomato':    {'sow_months': list(range(1,13)),        'harv_months': list(range(1,13)),     'min_dur': 55,  'max_dur': 130, 'min_drop': 0.20, 'min_rise': 0.15},
    'Paddy':     {'sow_months': [5,6,7,8,9,10],               'harv_months': [9,10,11,12,1],        'min_dur': 90,  'max_dur': 180, 'min_drop': 0.20, 'min_rise': 0.15},
    'Sugarcane': {'sow_months': [12,1,2,3,4,5,6],               'harv_months': [10,11,12,1,2,3,4,5],          'min_dur': 180, 'max_dur': 420, 'min_drop': 0.12, 'min_rise': 0.12},
    'Cotton':    {'sow_months': [5,6,7,8,9,10,11,12],             'harv_months': [11,12,1,2,3,4,5],           'min_dur': 65,  'max_dur': 230, 'min_drop': 0.10, 'min_rise': 0.10},
    'Gram':      {'sow_months': [9,10,11,12,1],          'harv_months': [1,2,3,4,5],           'min_dur': 70,  'max_dur': 180, 'min_drop': 0.10, 'min_rise': 0.10},
    'Lentil':    {'sow_months': [9,10,11,12,1],          'harv_months': [1,2,3,4,5],           'min_dur': 70,  'max_dur': 180, 'min_drop': 0.10, 'min_rise': 0.10},
    'Maize':     {'sow_months': [4,5,6,7,8,9,10,11],    'harv_months': [7,8,9,10,11,12,1,2],  'min_dur': 60,  'max_dur': 180, 'min_drop': 0.12, 'min_rise': 0.12},
    'Mustard':   {'sow_months': [8,9,10,11,12],          'harv_months': [1,2,3,4],             'min_dur': 70,  'max_dur': 180, 'min_drop': 0.10, 'min_rise': 0.10},
    'Potato':    {'sow_months': [9,10,11,12,1,2],        'harv_months': [1,2,3,4,5],           'min_dur': 60,  'max_dur': 150, 'min_drop': 0.10, 'min_rise': 0.10},
    'Rice':      {'sow_months': [4,5,6,7,8],             'harv_months': [8,9,10,11,12,1],      'min_dur': 70,  'max_dur': 210, 'min_drop': 0.12, 'min_rise': 0.12},
    'Wheat':     {'sow_months': [9,10,11,12,1],          'harv_months': [2,3,4,5,6],           'min_dur': 80,  'max_dur': 200, 'min_drop': 0.10, 'min_rise': 0.10},
    'Banana':    {'sow_months': list(range(1,13)),        'harv_months': list(range(1,13)),     'min_dur': 150, 'max_dur': 500, 'min_drop': 0.03, 'min_rise': 0.03},
    'Grapes':    {'sow_months': list(range(1,13)),        'harv_months': list(range(1,13)),     'min_dur': 45,  'max_dur': 350, 'min_drop': 0.03, 'min_rise': 0.03},
    'Soybean':   {'sow_months': [5,6,7,8],               'harv_months': [8,9,10,11,12],        'min_dur': 70,  'max_dur': 180, 'min_drop': 0.12, 'min_rise': 0.12},
    'Jowar':     {'sow_months': [5,6,7,8,9,10,11],       'harv_months': [8,9,10,11,12,1,2,3],  'min_dur': 70,  'max_dur': 180, 'min_drop': 0.10, 'min_rise': 0.10},
    'Bajra':     {'sow_months': [5,6,7,8],               'harv_months': [8,9,10,11],           'min_dur': 60,  'max_dur': 150, 'min_drop': 0.10, 'min_rise': 0.10},
    'Tur':       {'sow_months': [5,6,7,8,9],             'harv_months': [11,12,1,2,3,4],       'min_dur': 120, 'max_dur': 320, 'min_drop': 0.10, 'min_rise': 0.10},
    'Turmeric':  {'sow_months': [4,5,6,7,8],             'harv_months': [12,1,2,3,4,5],        'min_dur': 150, 'max_dur': 350, 'min_drop': 0.03, 'min_rise': 0.03},
    'Pomegranate':{'sow_months': list(range(1,13)),       'harv_months': list(range(1,13)),     'min_dur': 90,  'max_dur': 450, 'min_drop': 0.03, 'min_rise': 0.03},
    'Safflower': {'sow_months': [9,10,11,12],             'harv_months': [2,3,4,5],            'min_dur': 80,  'max_dur': 200, 'min_drop': 0.10, 'min_rise': 0.10},
    'Cabbage':   {'sow_months': list(range(1,13)),        'harv_months': list(range(1,13)),     'min_dur': 45,  'max_dur': 150, 'min_drop': 0.08, 'min_rise': 0.08},
}

CROP_RULES = RULES
CROP_TIMELINES = RULES

try:
    _trapz = np.trapezoid   # NumPy >= 2.0
except AttributeError:
    _trapz = np.trapz       # NumPy < 2.0


# ── Core functions ────────────────────────────────────────────────────────────

def split_into_years(dates):
    start = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    segments = []
    seg_start = start
    while seg_start <= end:
        seg_end = seg_start + pd.DateOffset(years=1)
        segments.append((seg_start, seg_end))
        seg_start = seg_end
    return segments


def validate_crop(crop, sow_date, peak_date, harvest_date, ndvi_curve):
    """Return True only when the lifecycle satisfies the crop's agronomic rules."""
    rule = RULES.get(crop)
    if rule is None:
        return False

    if sow_date is None or peak_date is None or harvest_date is None:
        return False

    sow_date = pd.Timestamp(sow_date)
    peak_date = pd.Timestamp(peak_date)
    harvest_date = pd.Timestamp(harvest_date)

    if not (sow_date <= peak_date <= harvest_date):
        return False

    sow_month = sow_date.month
    harvest_month = harvest_date.month
    if sow_month not in rule.get('sow_months', []):
        return False
    if harvest_month not in rule.get('harv_months', []):
        return False

    duration = (harvest_date - sow_date).days
    if duration < rule.get('min_dur', 0) or duration > rule.get('max_dur', float('inf')):
        return False

    curve = np.asarray(ndvi_curve, dtype=float)
    if curve.size < 3:
        return False

    peak_value = float(np.max(curve))
    sow_value = float(curve[0])
    harvest_value = float(curve[-1])
    rise = peak_value - sow_value
    drop = peak_value - harvest_value

    if rise < rule.get('min_rise', 0.0):
        return False
    if drop < rule.get('min_drop', 0.0):
        return False

    return True


def _has_large_gap_in_span(dates, start_idx, end_idx, max_gap_days=75):
    if start_idx >= end_idx:
        return False
    span_dates = pd.to_datetime(np.asarray(dates[start_idx:end_idx + 1]))
    gaps = pd.Series(span_dates).diff().dt.days.dropna()
    return bool((gaps > max_gap_days).any())


def _local_min_indices(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 3:
        return np.array([], dtype=int)
    mins = np.arange(1, len(values) - 1)
    mins = mins[(values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])]
    return mins.astype(int)


def find_valid_cycle_for_crop(crop, dates, ndvi, cfg, ref_date=None, candidate_cycles=None):
    """Select a lifecycle candidate that satisfies the crop's agronomic rules."""
    dates = np.asarray(dates)
    ndvi = np.asarray(ndvi, dtype=float)

    if candidate_cycles is None:
        candidate_cycles, _ = detect_cycles(dates, ndvi, cfg)

    if not candidate_cycles:
        return None

    valid_cycles = []
    for cycle in candidate_cycles:
        sow_idx = int(cycle.get('sowing', 0))
        peak_idx = int(cycle.get('peak', 0))
        harv_idx = int(cycle.get('harvest', len(ndvi) - 1))
        if sow_idx > peak_idx or peak_idx > harv_idx:
            continue
        if cycle.get('weak_boundary') and _has_large_gap_in_span(dates, sow_idx, harv_idx):
            continue
        if validate_crop(crop, dates[sow_idx], dates[peak_idx], dates[harv_idx], ndvi[sow_idx:harv_idx + 1]):
            valid_cycles.append({'sowing': sow_idx, 'peak': peak_idx, 'harvest': harv_idx})

    if not valid_cycles:
        return None

    if ref_date is not None:
        containing = [
            cycle for cycle in valid_cycles
            if pd.Timestamp(dates[cycle['sowing']]) <= pd.Timestamp(ref_date) <= pd.Timestamp(dates[cycle['harvest']])
        ]
        if containing:
            return containing[0]

    if ref_date is not None:
        return min(valid_cycles, key=lambda c: abs((pd.Timestamp(dates[c['peak']]) - pd.Timestamp(ref_date)).days))

    return valid_cycles[0]


def detect_cycles(dates, ndvi, cfg):
    """
    Notebook-compatible cycle detection.
    1) Smooth the NDVI series.
    2) Find peaks.
    3) For each peak, search left and right window bounds for the minimum
       smoothed NDVI value around the peak, matching the notebook logic.
    """
    s = pd.Series(ndvi).rolling(cfg['smooth'], center=True, min_periods=1).mean().values
    max_cycles = cfg['max_cycles']
    min_dist = max(3, len(s) // (max_cycles * 3))

    if cfg['use_prominence']:
        peaks, _ = find_peaks(s, prominence=cfg['prominence'], distance=min_dist)
    else:
        peaks, _ = find_peaks(s, height=0.3, distance=min_dist)

    if len(peaks) == 0:
        return [], s

    if len(peaks) > max_cycles:
        top_idx = np.argsort(s[peaks])[-max_cycles:]
        peaks = np.sort(peaks[top_idx])

    search_window = pd.DateOffset(months=cfg['search_months'])
    cycles = []
    local_minima = _local_min_indices(s)

    for p in peaks:
        peak_date = pd.Timestamp(dates[p])
        weak_boundary = False

        before_mask = (dates >= peak_date - search_window) & (dates <= peak_date)
        before_idx = np.where(before_mask)[0]
        if len(before_idx) > 0:
            left_candidates = before_idx[before_idx < p]
            if len(left_candidates) > 0:
                left_minima = left_candidates[np.isin(left_candidates, local_minima)]
                if len(left_minima) > 0:
                    sowing = int(left_minima[-1])
                else:
                    sowing = int(left_candidates[np.argmin(s[left_candidates])])
                    weak_boundary = True
            else:
                sowing = 0
        else:
            sowing = 0

        after_mask = (dates >= peak_date) & (dates <= peak_date + search_window)
        after_idx = np.where(after_mask)[0]
        if len(after_idx) > 0:
            right_candidates = after_idx[after_idx > p]
            if len(right_candidates) > 0:
                right_minima = right_candidates[np.isin(right_candidates, local_minima)]
                if len(right_minima) > 0:
                    harvest = int(right_minima[0])
                else:
                    harvest = int(right_candidates[np.argmin(s[right_candidates])])
                    weak_boundary = True
            else:
                harvest = int(after_idx[np.argmin(s[after_idx])]) if len(after_idx) > 0 else len(s) - 1
        else:
            harvest = len(s) - 1

        cycles.append({'sowing': sowing, 'peak': p, 'harvest': harvest, 'weak_boundary': weak_boundary})

    return cycles, s


def build_features_from_cycle(dates, ndvi, smooth, c):
    sow_i, peak_i, harv_i = c['sowing'], c['peak'], c['harvest']
    sow_date, peak_date, harv_date = dates[sow_i], dates[peak_i], dates[harv_i]

    seg_raw = ndvi[sow_i:harv_i + 1]
    seg_smooth = smooth[sow_i:harv_i + 1]
    if len(seg_raw) < 3:
        return None

    seg_days  = (dates[sow_i:harv_i + 1] - dates[sow_i]).astype('timedelta64[D]').astype(float)
    duration  = (pd.Timestamp(harv_date) - pd.Timestamp(sow_date)).days
    rise_days = max((pd.Timestamp(peak_date) - pd.Timestamp(sow_date)).days, 1)
    fall_days = max((pd.Timestamp(harv_date) - pd.Timestamp(peak_date)).days, 1)

    feats = {
        'Duration_Days': duration,
        'Peak_NDVI':     float(smooth[peak_i]),
        'Peak_Month':    pd.Timestamp(peak_date).month,
        'Sowing_NDVI':   float(smooth[sow_i]),
        'Sowing_Month':  pd.Timestamp(sow_date).month,
        'Harvest_NDVI':  float(smooth[harv_i]),
        'Harvest_Month': pd.Timestamp(harv_date).month,
        'AUC':           float(_trapz(seg_smooth, x=seg_days)) if len(seg_smooth) > 1 else 0.0,
        'Mean_NDVI':     float(np.mean(seg_raw)),
        'Std_NDVI':      float(np.std(seg_raw)),
        'NDVI_range':    float(np.ptp(seg_raw)),
        'Rise_rate':     round((smooth[peak_i] - smooth[sow_i]) / rise_days, 5),
        'Fall_rate':     round((smooth[peak_i] - smooth[harv_i]) / fall_days, 5),
    }
    return feats, sow_date, peak_date, harv_date


