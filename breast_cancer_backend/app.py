from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os


app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Load the saved model
try:
    model = tf.keras.models.load_model('breast_cancer_model_mobilenet.h5')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

def preprocess_image(img_path):
    """Loads, preprocesses, and prepares the image for the model."""
    img = image.load_img(img_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = img_array / 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    try:
        # Save the uploaded image temporarily
        temp_path = "temp_image.png"
        file.save(temp_path)

        # Preprocess the image
        img_array = preprocess_image(temp_path)

        # Make prediction
        prediction = model.predict(img_array)[0][0]

        # Determine the result and risk score
        risk_score = float(prediction)
        if prediction > 0.5:
            result = "Malignant" 
        else:
            result = "Benign"

        # Clean up the temporary image
        os.remove(temp_path)

        return jsonify({'result': result, 'risk_score': risk_score})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)