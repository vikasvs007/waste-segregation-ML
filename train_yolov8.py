from ultralytics import YOLO

# Load the YOLOv8 model for classification
model = YOLO('yolov8n-cls.pt')  # You can use 'yolov8s-cls.pt', 'yolov8m-cls.pt', etc., for larger models

# Train the model on your dataset
model.train(
    data='C:/Users/vikas/OneDrive/Desktop/Sustainable Management hackthon/waste_data.yaml',  # Path to the YAML file with dataset info
    epochs=10,               # Number of epochs to train
    imgsz=128,               # Image size (consistent with preprocessing)
    batch=16                 # Batch size (adjust based on available memory)
)
