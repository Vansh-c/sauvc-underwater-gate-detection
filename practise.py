import cv2
import time
from ultralytics import YOLO
from imutils.video import VideoStream

model = YOLO("best.pt", task="obb")

vs = VideoStream(src=0).start()
time.sleep(2.0)

prev_time = 0

pattern = [1,1,0]   # detect, detect, skip (average interval ≈ 1.5)
pattern_index = 0

results = None
last_annotated = None

while True:

    frame = vs.read()
    if frame is None:
        continue

    frame = cv2.resize(frame,(640,480))

    # Pattern-based detection
    if pattern[pattern_index] == 1:

        results = model.predict(
            frame,
            imgsz=384,
            conf=0.7,
            verbose=False
        )

        if results is not None:

            last_annotated = results[0].plot()

            if results[0].obb is not None:

                boxes = results[0].obb.xyxyxyxy.cpu().numpy()

                for box in boxes:

                    pts = box.reshape(4,2)

                    cx = int(pts[:,0].mean())
                    cy = int(pts[:,1].mean())

                    cv2.circle(last_annotated,(cx,cy),8,(0,0,255),-1)

    pattern_index = (pattern_index + 1) % len(pattern)

    # Reuse last detection
    if last_annotated is not None:
        annotated = last_annotated.copy()
    else:
        annotated = frame

    # FPS calculation
    current_time = time.time()
    fps = 1/(current_time-prev_time) if prev_time>0 else 0
    prev_time = current_time

    cv2.putText(
        annotated,
        f"FPS:{int(fps)}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0,255,255),
        2
    )

    cv2.imshow("Detection", annotated)

    if cv2.waitKey(1) & 0xFF == 27:
        break

vs.stop()
cv2.destroyAllWindows()