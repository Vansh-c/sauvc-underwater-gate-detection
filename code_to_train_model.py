from ultralytics import YOLO

model = YOLO("yolov8n-obb.pt")

model.train(
    data=r"E:\underwater_gate_detection(final)\underwater_gate_detection.v2i.yolov8-obb\data.yaml",
    task="obb",
    epochs=80,            # a bit higher since CPU is slower but stable
    imgsz=512,
    batch=4,              # safest for CPU RAM
    workers=2,            # Windows-friendly
    device="cpu",         # ✅ CPU ONLY
    patience=20,          # early stopping
    pretrained=True,
    name="gate_detection_model_obb_cpu"
)