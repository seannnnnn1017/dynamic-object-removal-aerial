from ultralytics import YOLO

# Load model
model = YOLO(r'E:\論文\期刊\code\YOLO\satellite3_train.pt')

# Run prediction and explicitly specify show and save
model.predict(r'E:\論文\期刊\code\final_video\test5_480fps_input.mp4', show=True, save=True, conf=0.15)
