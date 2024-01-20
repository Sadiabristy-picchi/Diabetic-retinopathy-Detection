from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np
import logging

app = Flask(__name__)
model = tf.keras.models.load_model(r'C:\DR1\venv\resnet_model_after_dcgan.h5')

# Configure logging
logging.basicConfig(level=logging.INFO)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})

        # Validate file type (optional)
        allowed_extensions = {'png', 'jpg', 'jpeg'}
        if '.' not in file.filename or file.filename.rsplit('.', 1)[1].lower() not in allowed_extensions:
            return jsonify({'error': 'Invalid file type. Allowed types: png, jpg, jpeg'})

        # Preprocess the image
        img = Image.open(file)
        img = img.resize((128, 128))  # Assuming your model expects input size 128x128
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)

        # Extract predicted class
        predicted_class = np.argmax(prediction, axis=1)

        # Map class index to corresponding class name
        class_names = [
            "No Diabetic retinopathy",
            "Mild Non-Proliferative Retinopathy",
            "Moderate Non-Proliferative Retinopathy",
            "Severe Non-Proliferative Retinopathy",
            "Proliferative Diabetic Retinopathy (PDR)"
        ]

        return jsonify({'predicted_class': class_names[predicted_class[0]]})

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({'error': 'An error occurred during prediction'})

if __name__ == '__main__':
    app.run(debug=True)
