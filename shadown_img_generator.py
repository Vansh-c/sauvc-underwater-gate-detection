import cv2
import numpy as np
import os
import random

VIDEO_PATH = r"E:\weird_projects\training_yolo_model\WhatsApp Video 2026-02-22 at 13.20.00.mp4"
OUTPUT_DIR = r"E:\weird_projects\training_yolo_model"

def add_shadow_texture(img):
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    for _ in range(random.randint(1, 3)):
        x1, y1 = random.randint(0, w), random.randint(0, h)
        x2, y2 = random.randint(0, w), random.randint(0, h)
        thickness = random.randint(120, 300)
        cv2.line(mask, (x1, y1), (x2, y2), 255, thickness)

    mask = cv2.GaussianBlur(mask, (201, 201), 0)
    strength = random.uniform(0.3, 0.6)
    mask = (mask / 255.0) * strength

    shaded = img.astype(float)
    for c in range(3):
        shaded[:, :, c] *= (1 - mask)

    return shaded.astype(np.uint8)

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # fallback

frame_interval = int(fps / 2)  # 2 frames per second

frame_id = 0
saved_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_id % frame_interval == 0:
        shadow_img = add_shadow_texture(frame)
        out_name = f"shadow_frame_{saved_id:04d}.jpg"
        cv2.imwrite(os.path.join(OUTPUT_DIR, out_name), shadow_img)
        saved_id += 1

    frame_id += 1

cap.release()
print(f"✅ Done. Generated {saved_id} non-identical shadowy images at 2 FPS.")