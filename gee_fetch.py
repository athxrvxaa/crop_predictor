"""
gee_fetch.py
------------
Fetches Sentinel-2 NDVI time series from Google Earth Engine for a point.
Requires: pip install earthengine-api
Run `earthengine authenticate` once before first use.
"""

import pandas as pd

def init_ee(project: str = 'gee-project-497010'):
    import ee
    try:
        ee.Initialize(project=project)
    except Exception:
        ee.Authenticate()
        ee.Initialize(project=project)


def get_ndvi_timeseries(lat: float, lon: float,
                        start_date: str, end_date: str,
                        cloud_thresh: int = 20,
                        project: str = 'gee-project-497010') -> pd.DataFrame:
    import ee
    init_ee(project)

    point = ee.Geometry.Point([lon, lat])

    def mask_and_ndvi(img):
        scl = img.select('SCL')
        valid_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        ndvi = ndvi.updateMask(valid_mask)
        ndvi = ndvi.updateMask(ndvi.gte(0))
        return img.addBands(ndvi)

    coll = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
            .filterBounds(point)
            .filterDate(start_date, end_date)
            .map(mask_and_ndvi))

    def extract(img):
        val = img.select('NDVI').reduceRegion(ee.Reducer.mean(), point, 10).get('NDVI')
        return ee.Feature(None, {'date': img.date().format('YYYY-MM-dd'), 'NDVI': val})

    feats = coll.map(extract).filter(ee.Filter.notNull(['NDVI']))
    data  = feats.reduceColumns(
        ee.Reducer.toList(2), ['date', 'NDVI']
    ).get('list').getInfo()

    df = pd.DataFrame(data, columns=['Date', 'NDVI'])
    if df.empty:
        return df
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').groupby('Date', as_index=False).mean()
    return df.reset_index(drop=True)
