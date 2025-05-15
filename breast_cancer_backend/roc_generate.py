import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import os
import cv2

# Load the saved model
model = tf.keras.models.load_model('breast_cancer_model_mobilenet.h5')

# Load your test data
data_dir = '/Users/safuraminhaj/mias_sorted'
image_size = (224, 224)
X_test = []
y_test = []

normal_dir = os.path.join(data_dir, 'benign')
for filename in os.listdir(normal_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(normal_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
            img = img / 255.0
            X_test.append(img)
            y_test.append(0)

abnormal_dir = os.path.join(data_dir, 'malignant')
for filename in os.listdir(abnormal_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(abnormal_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, image_size)
            img = np.expand_dims(img, axis=-1)
            img = np.repeat(img, 3, axis=-1)
            img = img / 255.0
            X_test.append(img)
            y_test.append(1)

X_test = np.array(X_test)
y_test = np.array(y_test)

# Make predictions on the test set to get probability scores
y_pred_proba = model.predict(X_test).ravel()

# Calculate the ROC curve to get FPR and TPR at various thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

# Calculate TNR and FNR
tnr = 1 - fpr
fnr = 1 - tpr

# Plot the TNR vs FNR curve
plt.figure(figsize=(8, 8))
plt.plot(fnr, tnr, color='darkgreen', lw=2, label='TNR vs FNR Curve')
plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--') # Diagonal line for random classifier
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Negative Rate (FNR)')
plt.ylabel('True Negative Rate (TNR)')
plt.title('True Negative Rate vs. False Negative Rate Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()

# You can also calculate an "Area Under the TNR-FNR Curve" (AUCN)
# which would be equivalent to the standard AUC.
aucn = roc_auc_score(1 - y_test, 1 - y_pred_proba) # Invert labels and probabilities
print(f"Area Under the TNR-FNR Curve (AUCN): {aucn:.2f}")

# Here's what that means in the context of your breast cancer detection model and the right-angle-ish ROC curve:

# Very High Discriminative Power: An AUC of 0.99 indicates that your model has a very high ability to distinguish between the "Malignant" and "Benign" classes. There's only a 1% chance that a randomly chosen malignant case will be ranked lower by your model than a randomly chosen benign case.

# Alignment with the Right-Angle Curve: Your description of the ROC curve looking like a right angle in the top-left corner perfectly aligns with a very high AUC.

# The steep rise to a high True Positive Rate (TPR) at a very low False Positive Rate (FPR) contributes to a large area under the curve.
# The curve staying high (close to TPR = 1) for a while before potentially increasing the FPR further maximizes this area.
# In practical terms for your application:

# High Sensitivity and Specificity: A high AUC suggests that you can likely find a probability threshold where your model achieves both high sensitivity (correctly identifying most malignant cases) and high specificity (correctly identifying most benign cases), with a minimal trade-off.
# Potentially Very Useful Tool: This level of performance suggests your model could be a valuable tool in assisting with breast cancer diagnosis, assuming it generalizes well to unseen data.
# Important Considerations (Even with a High AUC):

# Generalization to Unseen Data: While a high AUC on your test set is promising, it's crucial to ensure that the model generalizes well to completely new, unseen data. Overfitting to the training or test set could lead to a drop in performance on real-world data.
# Clinical Context: The model's predictions should always be interpreted within the clinical context and by medical professionals. It should serve as an aid, not a replacement for expert judgment.
# Class Imbalance: If your original dataset had a significant class imbalance, a high AUC is still good, but you might want to look at other metrics like precision, recall, and F1-score to get a more complete picture of performance, especially for the minority class.
# In summary, an AUC of 0.99 is a very strong result, and the right-angle shape of your ROC curve supports this. It indicates a highly effective model for distinguishing between benign and malignant breast tissue in your dataset.