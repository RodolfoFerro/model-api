# ===============================================================
# Author: Rodolfo Ferro
# Email: ferro@cimat.mx
# Twitter: @rodo_ferro
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro.
# Any explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ===============================================================

# -*- coding: utf-8 -*-

from flask import Flask
from flask import jsonify
from flask import request
import tensorflow as tf
import numpy as np

# Flask app
app = Flask(__name__)

# Globals
VERSION = 'v0.0'
BASE_URL = f'/api/{VERSION}/'
SPECIES = {0: 'I. setosa', 1: 'I. versicolor', 2: 'I. virginica'}

# Load trained model
MODEL_PATH = "models/iris-model.keras"
model = tf.keras.models.load_model(MODEL_PATH)


# API Routes
@app.route(BASE_URL, methods=['GET'])
def test():
    """
    GET method to test the API.
    """

    # Output message:
    message = {"response": [{"text": "Hello world!"}]}
    return jsonify(message)


@app.route(BASE_URL + '/predict', methods=['POST'])
def predict():
    """
    POST method to predict with our classification model.
    """

    # Get data from JSON object in POST method
    req_data = request.get_json()

    # Parse data from JSON input
    sl = req_data['sepal_length']
    sw = req_data['sepal_width']
    pl = req_data['petal_length']
    pw = req_data['petal_width']

    # Perform inference
    input_data = np.array([[sl, sw, pl, pw]])
    prediction = model.predict(input_data)
    label = np.argmax(prediction)
    class_name = SPECIES[label]
    print(f'[INFO] Predicted class: {class_name}')

    # Build output message
    message = {"response": [
        {"input": {
            'sepal_length': sl,
            'sepal_width': sw,
            'petal_length': pl,
            'petal_width': pw
        }},
        {"prediction": int(label)},
        {"species": str(class_name)}]}

    return jsonify(message)


@app.errorhandler(404)
def not_found(error=None):
    """
    Route for 404.
    """

    print('[INFO] Error:', error)

    message = {
        'status': 404,
        'message': 'Not Found: ' + request.url,
    }

    response = jsonify(message)
    response.status_code = 404

    return response


if __name__ == '__main__':
    app.run(debug=True, port=5000)
