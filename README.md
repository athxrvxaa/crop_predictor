# Crop Predictor — VS Code Setup

## Folder structure
```
crop_predictor/
├── app.py              ← Flask web server
├── train.py            ← Train model from CSV → saves model/crop_classifier.pkl
├── predict.py          ← Prediction logic (used by app.py)
├── pipeline.py         ← Core lifecycle detection + features
├── gee_fetch.py        ← GEE NDVI fetcher
├── requirements.txt
├── model/              ← pkl saved here after training
└── templates/
    └── index.html      ← Web UI
```

---

## Step 1 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Step 2 — Authenticate Google Earth Engine (one time only)

```bash
earthengine authenticate
```

This opens a browser. Login with your Google account that has GEE access.

---

## Step 3 — Train the model

Put your CSV (`all_5_crops_combined.csv`) in the `crop_predictor/` folder, then:

```bash
python train.py --csv all_5_crops_combined.csv
```

This runs the full pipeline (detection → validation → features → RF) and saves:
```
model/crop_classifier.pkl
```

---

## Step 4 — Run the web app

```bash
python app.py
```

Open http://localhost:5000 in your browser.

---

## Using the UI

1. Enter **Latitude** and **Longitude** of the farm point
2. Enter the **Ground Truth Date** (date of field visit)
3. Optionally enter the actual crop name to check if prediction is correct
4. Click **Predict Crop**

The app will:
- Fetch Sentinel-2 NDVI live from GEE for ±8 months around the date
- Detect the lifecycle (sowing → peak → harvest)
- Predict the crop using the trained Random Forest
- Show the NDVI chart with lifecycle events marked

---

## GEE Project

Default project: `gee-project-497010`

To change it, edit this line in `gee_fetch.py`:
```python
def init_ee(project: str = 'gee-project-497010'):
```
