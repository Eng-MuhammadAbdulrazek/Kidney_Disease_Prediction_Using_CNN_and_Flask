# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2

# Some utilities
import numpy as np
from util import base64_to_pil

# Declare a flask app
app = Flask(__name__)

# model = MobileNetV2(weights='imagenet')

# Model saved with Keras model.save()
MODEL_PATH = 'models/KidneyDiseasesModel.h5'

model = load_model(MODEL_PATH)

model.make_predict_function()  # Necessary

print('loading model & Start serving...')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img, Model):
    img = img.resize((200, 200))

    img_gray = img.convert('L')

    test_image = image.img_to_array(img_gray)

    test_image = np.expand_dims(test_image, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    # x = preprocess_input(test_image, mode='tf')

    preds = Model.predict(test_image)
    return preds


def determine_case(x):
    match x:
        case 0:
            return "Cyst"
        case 1:
            return "Normal"
        case 2:
            return "Stone"
        case 3:
            return "Tumor"


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        print('----------------------------------------------------')
        print('incoming request :')
        # Get the image from post request
        img = base64_to_pil(request.json['data'])
        # Making the prediction
        prediction = model_predict(img, model)[0]
        index_of_max = np.argmax(prediction)
        case = determine_case(index_of_max)
        print('prediction : ', case)
        print('----------------------------------------------------')
        # Serialize the result, you can add additional fields
        return jsonify(result=case)

    return None


if __name__ == '__main__':
    # app.run(port=5002, threaded=False)

    # Serve the app with gevent
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
