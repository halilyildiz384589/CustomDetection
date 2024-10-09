from ultralytics import YOLO

model = YOLO("yolov8m.pt")

model.train(data="config.yaml", batch=8, imgsz=640, epochs=100, workers=1)