import argparse
import numpy as np
import pandas as pd

from pipeline import CROP_CONFIG, RULES, detect_cycles, validate_crop


def _local_minimum(value_series, idx):
    if idx <= 0 or idx >= len(value_series) - 1:
        return False
    s = value_series
    return s[idx] <= s[idx - 1] and s[idx] <= s[idx + 1]


def debug_cycle(lat, lon, ref_date, window_months=12, project='gee-project-497010'):
    from gee_fetch import get_ndvi_timeseries

    ref = pd.Timestamp(ref_date)
    start = (ref - pd.DateOffset(months=window_months)).strftime('%Y-%m-%d')
    end = (ref + pd.DateOffset(months=window_months)).strftime('%Y-%m-%d')

    ndvi_df = get_ndvi_timeseries(lat, lon, start, end, project=project)
    if ndvi_df.empty:
        print('No observations found')
        return

    dates = pd.to_datetime(ndvi_df['Date']).to_numpy(dtype='datetime64[ns]')
    ndvi = ndvi_df['NDVI'].to_numpy(dtype=float)
    cycles, smooth = detect_cycles(dates, ndvi, CROP_CONFIG.get('Sugarcane', {'smooth': 3, 'max_cycles': 1, 'search_months': 6, 'use_prominence': False, 'prominence': None}))

    print(f'obs_count={len(ndvi_df)}')
    if len(ndvi_df) > 1:
        gaps = pd.Series(pd.to_datetime(ndvi_df['Date'])).diff().dt.days.dropna()
        print(f'largest_gap_days={int(gaps.max()) if not gaps.empty else 0}')
    else:
        print('largest_gap_days=0')
    print('smoothed=', [round(float(v), 4) for v in smooth])

    for idx, cycle in enumerate(cycles):
        sow = int(cycle['sowing'])
        peak = int(cycle['peak'])
        harvest = int(cycle['harvest'])
        sow_date = pd.Timestamp(dates[sow])
        peak_date = pd.Timestamp(dates[peak])
        harv_date = pd.Timestamp(dates[harvest])
        print(f'cycle[{idx}] sow={sow}({sow_date.date()}) peak={peak}({peak_date.date()}) harvest={harvest}({harv_date.date()})')
        print(f'  boundary_local_minimum: sow={_local_minimum(smooth, sow)} harvest={_local_minimum(smooth, harvest)}')

    for crop in sorted(RULES.keys()):
        cfg = CROP_CONFIG.get(crop, {'smooth': 3, 'max_cycles': 1, 'search_months': 6, 'use_prominence': False, 'prominence': None})
        crop_cycles, _ = detect_cycles(dates, ndvi, cfg)
        failed = []
        for c in crop_cycles:
            sow = int(c['sowing'])
            peak = int(c['peak'])
            harv = int(c['harvest'])
            sow_date = pd.Timestamp(dates[sow])
            peak_date = pd.Timestamp(dates[peak])
            harv_date = pd.Timestamp(dates[harv])
            curve = ndvi[sow:harv + 1]
            if not validate_crop(crop, sow_date, peak_date, harv_date, curve):
                rule = RULES[crop]
                reasons = []
                if sow_date.month not in rule['sow_months']:
                    reasons.append('sow_month')
                if harv_date.month not in rule['harv_months']:
                    reasons.append('harv_month')
                if not (rule['min_dur'] <= (harv_date - sow_date).days <= rule['max_dur']):
                    reasons.append('duration')
                if (curve.max() - curve[-1]) < rule.get('min_drop', 0):
                    reasons.append('drop')
                if (curve.max() - curve[0]) < rule.get('min_rise', 0):
                    reasons.append('rise')
                failed.append((sow_date.date(), peak_date.date(), harv_date.date(), reasons))
        if failed:
            print(f'crop={crop} rejected={failed}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Debug lifecycle detection for a lat/lon/date.')
    parser.add_argument('lat', type=float)
    parser.add_argument('lon', type=float)
    parser.add_argument('date', type=str)
    parser.add_argument('--window-months', type=int, default=12)
    parser.add_argument('--project', type=str, default='gee-project-497010')
    args = parser.parse_args()
    debug_cycle(args.lat, args.lon, args.date, args.window_months, args.project)
