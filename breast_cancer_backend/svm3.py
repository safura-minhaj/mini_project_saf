import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import os
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.utils import class_weight
from tensorflow.keras.preprocessing import image

# 1. Data Loading and Preprocessing
data_dir = '/Users/safuraminhaj/mias_sorted'
image_size = (224, 224)

images = []
labels = []

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
            images.append(img)
            labels.append(0)
        else:
            print(f"Warning: Failed to load image {filename} in normal folder.")

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
            images.append(img)
            labels.append(1)
        else:
            print(f"Warning: Failed to load image {filename} in abnormal folder.")

images = np.array(images)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Check for class imbalance and calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: weight for i, weight in enumerate(class_weights)}

datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(X_train)

# 2. CNN Model (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)  # Add dropout
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

optimizer = Adam(learning_rate=0.0001)  # Adjusted learning rate
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 3. Model Training
model.fit(
    datagen.flow(X_train, y_train, batch_size=16),  # Adjusted batch size
    epochs=20,  # Increased epochs
    validation_data=(X_test, y_test),
    class_weight=class_weights
)

# 4. Model Evaluation
model.evaluate(X_test, y_test)
# 5. Image Upload and Prediction Function
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224), color_mode='grayscale')
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.repeat(img_array, 3, axis=-1)
    img_array = img_array / 255.0

    prediction = model.predict(img_array)

    if prediction[0][0] > 0.5:
        print("Malignant")
    else:
        print("Benign")
#example usage of predict image for
# Example usage of predict_image (for a single image):
test_image_path = "/Users/safuraminhaj/mias_sorted/malignant/malignant (5).png"
test_image_path_1 = "/Users/safuraminhaj/mias_sorted/benign/benign (22).png"

 #replace with test image path.
prediction_result = predict_image(test_image_path)
prediction_result_2 = predict_image(test_image_path_1)
print(prediction_result,prediction_result_2)

# Save the model
model.save('breast_cancer_model_mobilenet.h5')