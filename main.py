from __future__ import division, print_function

# coding=utf-8
import os

# Flask utils
from flask import Flask, request, render_template
# Keras
from keras.applications.imagenet_utils import decode_predictions
from keras.models import load_model
from werkzeug.utils import secure_filename

import numpy as np

app = Flask(__name__, template_folder='template')

# Load your trained binary image classification model
model = load_model('imageclassifier.h5')
model.make_predict_function()

print('Model loaded.')


# Define a function to preprocess and classify the image
def model_predict(image):
    img = image.load_img(image, target_size=(200, 200))
    x = image.img_to_array(img)
    x = x / 255

    x = x.reshape(1, 200, 200, 3)

    preds = model.predict(x)
    # preds = np.argmax(preds, axis=1)
    res = preds[0]
    if res < 0.5:
        return 0
    else:
        return 1


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
