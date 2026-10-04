from ultralytics import YOLO

model = YOLO(r"E:\underwater_gate_detection(final)\runs\obb\gate_detection_model_obb_cpu\weights\best.pt")

model.predict(
    source=r"E:\underwater_gate_detection(final)\test_video.mp4",
    imgsz=512,
    conf=0.4,
    iou=0.6,
    save=True
)