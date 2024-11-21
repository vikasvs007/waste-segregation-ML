import os
import cv2
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Set paths and categories
base_path = r"C:\Users\vikas\OneDrive\Desktop\Sustainable Management hackthon\Dataset\train"  # Update this path
main_categories = [ "biodegradable","non_biodegradable"]

data, labels = [], []

# Load images from nested folders
for main_category in main_categories:
    main_path = os.path.join(base_path, main_category)
    label = main_categories.index(main_category)  # Numeric label for main category

    for subcategory in os.listdir(main_path):
        sub_path = os.path.join(main_path, subcategory)
        
        # Check if the path is a directory (subfolder)
        if os.path.isdir(sub_path):
            for img_name in os.listdir(sub_path):
                img_path = os.path.join(sub_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    # Resize and convert image to grayscale
                    img = cv2.resize(img, (128, 128))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    data.append(img)
                    labels.append(label)

# Convert data and labels to NumPy arrays
data = np.array(data)
labels = np.array(labels)
print("the array:",data)
print("the labels:",labels)

# Preprocess data and split into train/test sets
data_flat = [img.flatten() for img in data]
X_train, X_test, y_train, y_test = train_test_split(data_flat, labels, test_size=0.3, random_state=42)

# Train the SVM model
svm_model = svm.SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Evaluate model
y_pred = svm_model.predict(X_test)
print("the classification report\n :",classification_report(y_test, y_pred, target_names=main_categories))

# Save the trained model
model_path = "C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/waste_classifier_model.pkl"
joblib.dump(svm_model, model_path)
print(f"Model saved to {model_path}")

# Load and predict
svm_model = joblib.load("C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/waste_classifier_model.pkl")
print("Model loaded successfully.")

# Prediction function
def predict_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (128, 128))
    img_flat = img.flatten().reshape(1, -1)
    prediction = svm_model.predict(img_flat)
    return main_categories[prediction[0]]

# Test prediction
test_image_path = "C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/image.png"  # Update with an actual image path
print("Predicted class:", predict_image(test_image_path))
