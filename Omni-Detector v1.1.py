import cv2
from ultralytics import YOLO
import time
import os

MODEL_NAME = 'yolov8n.pt'  
CAMERA_INDEX = 0          
SAVE_SCREENSHOTS = True    
SCREENSHOT_FOLDER = "screenshots"

if SAVE_SCREENSHOTS and not os.path.exists(SCREENSHOT_FOLDER):
    os.makedirs(SCREENSHOT_FOLDER)

print("[🧠] Loading YOLOv8 model... (first run downloads ~50MB)")
model = YOLO(MODEL_NAME)
print("[✅] Model loaded successfully!")

CLASS_NAMES = model.names


cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("[❌] Error: Could not access webcam. Check connection or try index 1.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n[🎥] Omni-Detector Started!")
print("➡️  Press 'q' to QUIT")
print("➡️  Press 'c' to CAPTURE screenshot")
print("➡️  Press 't' to TOGGLE person-only mode")
print("➡️  Press 'a' to SHOW ALL classes again\n")

SHOW_ONLY_PERSON = False
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[⚠️] Failed to grab frame. Exiting...")
        break

    frame_count += 1

    if SHOW_ONLY_PERSON:
        results = model(frame, classes=[0]) 
    else:
        results = model(frame)  

    annotated_frame = results[0].plot()

    if frame_count % 10 == 0:
        detected_objects = []
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = CLASS_NAMES[cls_id]
            detected_objects.append(f"{label} ({conf:.2f})")

        if detected_objects:
            print(f"[🔍 Frame {frame_count}] Detected: {', '.join(detected_objects)}")
        else:
            print(f"[🔍 Frame {frame_count}] Nothing detected.")

    mode_text = "MODE: PERSON ONLY" if SHOW_ONLY_PERSON else "MODE: ALL OBJECTS"
    cv2.putText(annotated_frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("Omni-Detector v1.0 — Press 'q' to Quit", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'): 
        print("[🛑] User requested exit. Shutting down...")
        break

    elif key == ord('c'):  
        if SAVE_SCREENSHOTS:
            timestamp = int(time.time())
            filename = f"{SCREENSHOT_FOLDER}/capture_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"[📸] Screenshot saved: {filename}")
        else:
            print("[📸] Screenshot saving is disabled.")

    elif key == ord('t'):  
        SHOW_ONLY_PERSON = not SHOW_ONLY_PERSON
        mode = "Person-Only" if SHOW_ONLY_PERSON else "All Objects"
        print(f"[🔄] Toggled detection mode: {mode}")

    elif key == ord('a'): 
        SHOW_ONLY_PERSON = False
        print("[🔄] Detection mode: ALL OBJECTS")

cap.release()
cv2.destroyAllWindows()
print("[✅] Omni-Detector terminated. Goodbye!")