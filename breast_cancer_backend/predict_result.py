import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
model = tf.keras.models.load_model('breast_cancer_model_mobilenet.h5')

def predict_new_image_with_score(image_path):
    """
    Loads an image, preprocesses it, and makes a prediction using the loaded model,
    returning the risk score.

    Args:
        image_path (str): The path to the image file.

    Returns:
        tuple: A tuple containing the prediction label ('Malignant' or 'Benign')
               and the risk score (probability of being malignant).
    """
    img = image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)[0][0]  # Get the raw probability

    if prediction > 0.5:
        return "Malignant", prediction
    else:
        return "Benign", 1 - prediction

# The sigmoid activation function in the last layer of your model outputs a value between 0 and 1. This value can be interpreted as the probability of the input image belonging to the positive class, which in your case is "Malignant" (since you labeled it as 1)
# For a "Malignant" prediction with a high score (e.g., 0.85): This means the model is quite confident (85% probability) that the image shows malignant tissue based on the patterns it learned during training.
# For a "Malignant" prediction with a lower score (e.g., 0.6): This suggests the model leans towards "Malignant" but with less certainty (60% probability). There might be some features in the image that are less clearly indicative of malignancy.
# For a "Benign" prediction with a high score (e.g., 0.9): This means the model is highly confident (90% probability, or a 1 - 0.1 = 0.9 score in our implementation) that the image shows benign tissue.
# For a "Benign" prediction with a lower score (e.g., 0.55): This suggests the model thinks it's more likely "Benign" but with less strong evidence (55% probability of being malignant, hence 1 - 0.55 = 0.45 "benign risk" in our implementation).


if __name__ == '__main__':
    # Example usage of predict_new_image_with_score
    test_image_path = "/Users/safuraminhaj/mias_sorted/malignant/malignant (5).png"  # Replace with your image path
    prediction_label, risk_score = predict_new_image_with_score(test_image_path)
    print(f"Prediction for {test_image_path}: {prediction_label}, Risk Score: {risk_score:.4f}")

    test_image_path_2 = "/Users/safuraminhaj/mias_sorted/benign/benign (22).png" # Replace with another image path
    prediction_label_2, risk_score_2 = predict_new_image_with_score(test_image_path_2)
    print(f"Prediction for {test_image_path_2}: {prediction_label_2}, Risk Score: {risk_score_2:.4f}")

    # You can add more image paths here to test multiple images




    // src/App.js
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';
import rocCurveImage from './images/roc_curve.png'; // Import the ROC curve image

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [riskScore, setRiskScore] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [uploadedFileName, setUploadedFileName] = useState('');
  const [aucScore, setAucScore] = useState('0.99'); // Replace with your actual AUC score if known

  const handleImageChange = (event) => {
    const file = event.target.files[0];
    setSelectedImage(file);
    setUploadedFileName(file ? file.name : '');
    setPredictionResult(null);
    setRiskScore(null);
    setError('');
  };

  const handleUpload = async () => {
    if (!selectedImage) {
      setError('Please select an image.');
      return;
    }

    setLoading(true);
    setError('');
    setPredictionResult(null);
    setRiskScore(null);

    const formData = new FormData();
    formData.append('image', selectedImage);

    try {
      const response = await axios.post('http://localhost:5000/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      setPredictionResult(response.data.result);
      setRiskScore(parseFloat(response.data.risk_score).toFixed(4));
      setLoading(false);
    } catch (error) {
      console.error('Error during prediction:', error);
      setError('Failed to upload and predict the image.');
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Breast Cancer Prediction</h1>
        <p className="app-subheader">Upload an image for analysis and view model performance</p>
      </header>

      <main className="app-main">
        <div className="upload-section">
          <label htmlFor="image-upload" className="upload-label">
            Select Image
          </label>
          <input
            type="file"
            id="image-upload"
            accept="image/*"
            onChange={handleImageChange}
            className="upload-input"
          />
          {uploadedFileName && <p className="uploaded-file-name">Uploaded: {uploadedFileName}</p>}
          <button
            onClick={handleUpload}
            disabled={!selectedImage || loading}
            className="predict-button"
          >
            {loading ? 'Predicting...' : 'Predict'}
          </button>

          {error && <p className="error-message">{error}</p>}
        </div>

        {predictionResult && (
          <div className="prediction-results">
            <h2 className={`prediction-title ${predictionResult.toLowerCase()}`}>
              Prediction: {predictionResult}
            </h2>
            <p className="risk-score">Risk Score: {riskScore}</p>
            <div className="risk-indicator">
              <div
                className="risk-bar"
                style={{ width: `${riskScore * 100}%` }}
              ></div>
              <span className="risk-label">0</span>
              <span className="risk-label">1</span>
            </div>
          </div>
        )}

        <div className="roc-curve-section">
          <h2>ROC Curve</h2>
          <img src={rocCurveImage} alt="ROC Curve" className="roc-curve-image" />
          {aucScore && <p className="auc-score">AUC: {aucScore}</p>}
        </div>
      </main>

      <footer className="app-footer">
        <p>&copy; 2025 Breast Cancer Prediction App</p>
      </footer>
    </div>
  );
}

export default App;