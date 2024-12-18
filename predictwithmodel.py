# predict_with_model.py

import cv2
import joblib
import numpy as np

# Load the trained model
model_path = "C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/waste_classifier_model.pkl"
svm_model = joblib.load(model_path)
print("Model loaded successfully.",model_path)

# Define categories
main_categories = ["biodegradable", "non_biodegradable"]
print(main_categories)

# # Prediction function
# def predict_from_webcam(img):
#     # Initialize webcam capture
#     # url="http://192.0.0.4:8080/video"
#     # cap = cv2.VideoCapture(0)  # 0 is usually the default camera; adjust if needed

#     # print("Press 'q' to quit after capturing an image.")
#     # print("c to capture th photo:")
#     # while True:
#     #     ret, frame = cap.read()
#     #     if not ret:
#     #         print("Failed to capture image")
#     #         break

#     #     # Display the webcam feed
#     #     cv2.imshow("Webcam Feed", frame)
#     #     # Capture image when 'c' is pressed
#     #     if cv2.waitKey(1) & 0xFF == ord('c'):
#             # Preprocess the captured frame for prediction
#             # img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
#             # img = cv2.resize(img, (128, 128))  # Resize to the input size
#             img_flat = img.flatten().reshape(1, -1)  # Flatten and reshape for model input

#             # Make prediction
#             prediction = svm_model.predict(img_flat)
#             print("Predicted class:", main_categories[prediction[0]])

# #         # Exit loop when 'q' is pressed
# #         elif cv2.waitKey(1) & 0xFF == ord('q'):
# #             break
# #     cap.release()
# #     cv2.destroyAllWindows()
# #     # Release the webcam and close windows
# imagepath="C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/non.jpeg"

def predict_from_webcam(imagepath):
    img = cv2.imread(image_path)
    if img is None:
        print("Failed to load image. Check the image path.")
        return

    # Preprocess the image
    img = cv2.resize(img, (256, 256))  # Resize to model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img_flat = img.flatten().reshape(1, -1)   # Assuming extract_hog_features is defined

    # Make a prediction
    prediction = svm_model.predict(img_flat)  # Assuming svm_model is defined and loaded
    print("Predicted class:", main_categories[prediction[0]])

# Provide the path to the image you want to predict
image_path = "C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/Dataset/train/biodegradable/food_waste/0_338.jpg"
predict_from_webcam(image_path)
