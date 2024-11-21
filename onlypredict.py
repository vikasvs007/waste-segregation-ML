<<<<<<< HEAD



import cv2
import os
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# Specify the dataset path and categories
# Base path to the dataset
base_path = r"C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/Dataset/train"
  # Replace with your actual path

# Define main categories
main_categories = ["biodegradable", "non_biodegradable"]

# Initialize data and labels lists
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

def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    features = [hog.compute(img).flatten() for img in images]
    return np.array(features)

# Extract HOG features
data_hog = extract_hog_features(data)

X_train, X_test, y_train, y_test = train_test_split(data_hog, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=main_categories))


def predict_from_image(image_path):
    # Read the image from the provided path
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image. Check the image path.")
        return

    # Preprocess the image
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features = extract_hog_features([img])  # Assuming extract_hog_features is defined

    # Make a prediction
    prediction = svm.predict(features)  # Assuming svm_model is defined and loaded
    print("Predicted class:", main_categories[prediction[0]])

# Provide the path to the image you want to predict
image_path = "C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/image.png"
predict_from_image(image_path)
=======



import cv2
import os
import sklearn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
# Specify the dataset path and categories
# Base path to the dataset
base_path = r"C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/Dataset/train"
  # Replace with your actual path

# Define main categories
main_categories = ["biodegradable", "non_biodegradable"]

# Initialize data and labels lists
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

def extract_hog_features(images):
    hog = cv2.HOGDescriptor()
    features = [hog.compute(img).flatten() for img in images]
    return np.array(features)

# Extract HOG features
data_hog = extract_hog_features(data)

X_train, X_test, y_train, y_test = train_test_split(data_hog, labels, test_size=0.2, random_state=42)

# Train an SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=main_categories))


def predict_from_image(image_path):
    # Read the image from the provided path
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image. Check the image path.")
        return

    # Preprocess the image
    img = cv2.resize(img, (128, 128))  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    features = extract_hog_features([img])  # Assuming extract_hog_features is defined

    # Make a prediction
    prediction = svm.predict(features)  # Assuming svm_model is defined and loaded
    print("Predicted class:", main_categories[prediction[0]])

# Provide the path to the image you want to predict
image_path = "C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/image.png"
predict_from_image(image_path)
>>>>>>> 8baefeed6365a2c62d0efc3786f51b85a3df4943
