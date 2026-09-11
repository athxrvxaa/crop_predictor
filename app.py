"""
app.py
------
Flask web app — crop prediction UI.
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import traceback

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        lat          = float(data['lat'])
        lon          = float(data['lon'])
        ref_date     = data['ref_date']
        ground_truth = data.get('ground_truth', None) or None

        from predict import predict_crop
        result = predict_crop(lat, lon, ref_date, ground_truth=ground_truth)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


if __name__ == '__main__':
    print('Starting Crop Predictor — http://localhost:5000')
    app.run(debug=True, port=5000)
