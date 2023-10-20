from __future__ import division, print_function

# coding=utf-8
import os

# Flask utils
from flask import Flask, request, render_template
# Keras
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from werkzeug.utils import secure_filename
from keras.preprocessing.image import load_img, img_to_array


import numpy as np

app = Flask(__name__, template_folder='template')

# Load your trained binary image classification model
model = load_model('imageClassifier (1).h5')
model.make_predict_function()

print('Model loaded.')


# Define a function to preprocess and classify the image
def model_predict(image):
    img = load_img(image, target_size=(256, 256))
    x = img_to_array(img)
    x = x / 255

    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)
    # preds = np.argmax(preds, axis=1)
    # res = preds
    if preds < 0.5:
        preds = "Happy"
    else:
        preds = "Sad"

    return preds


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['files']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))

        f.save(file_path)

        # Make prediction
        # preds = model_predict(file_path)
        result = model_predict(file_path)

        # Process your result for human
        # pred_class = decode_predictions(preds, top=1)  # ImageNet Decode
        # result = str(pred_class[0][0][1])  # Convert to string
        return result
    return render_template('predict.html')


if __name__ == '__main__':
    app.run(debug=True)
