from flask import Flask, render_template, request, jsonify
import numpy as np
import cv2
import base64
from PIL import Image
from io import BytesIO
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('model.h5')

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = data['image'].split(",")[1]
    img_bytes = BytesIO(base64.b64decode(img_data))
    
    # Open image and convert to grayscale
    img = Image.open(img_bytes).convert('L')

    # Invert image so background becomes black, digit becomes white
    img = Image.eval(img, lambda x: 255 - x)

    # Resize to 28x28
    img = img.resize((28, 28))

    # Convert to numpy array
    img = np.array(img)

    # Normalize to [0, 1]
    img = img / 255.0

    # Reshape to match model input
    img = img.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img)[0]
    digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100

    return jsonify({"digit": digit, "confidence": round(confidence, 2)})

if __name__ == '__main__':
    app.run(debug=True)
