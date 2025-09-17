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

CONF_THRESHOLD = 0.45     
IOU_THRESHOLD = 0.7     
ENABLE_TRACKING = False   
TRACKING_CONF_THRESHOLD = 0.3 

print(f"[🧠] Loading YOLOv8 model '{MODEL_NAME}'... (first run downloads model)")
try:
    model = YOLO(MODEL_NAME)
    print("[✅] Model loaded successfully!")
except Exception as e:
    print(f"[❌] Error loading model: {e}")
    print("Please check your internet connection or model name.")
    exit()

CLASS_NAMES = model.names

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    print("[❌] Error: Could not access webcam. Check connection or try index 1.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\n[🎥] Omni-Detector v1.2 Started! (Optimized for Speed)")
print("-------------------------------------------------------")
print("➡️  Press 'q' to QUIT")
print("➡️  Press 'c' to CAPTURE screenshot")
print("➡️  Press 't' to TOGGLE person-only mode")
print("➡️  Press 'a' to SHOW ALL classes again")
print("➡️  Press 'k' to TOGGLE OBJECT TRACKING")
print("-------------------------------------------------------")
print(f"Current Model: {MODEL_NAME}. Consider 'yolov8s.pt' for a balance.")
print("For best performance, ensure you have a powerful GPU.")


SHOW_ONLY_PERSON = False
frame_count = 0
prev_frame_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("[⚠️] Failed to grab frame. Exiting...")
        break

    frame_count += 1
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    inference_args = {
        'conf': CONF_THRESHOLD,
        'iou': IOU_THRESHOLD,
        'classes': [0] if SHOW_ONLY_PERSON else None, 
        'half': True 
    }

    if ENABLE_TRACKING:
        tracking_args = inference_args.copy()
        tracking_args['conf'] = TRACKING_CONF_THRESHOLD
        results = model.track(frame, persist=True, **tracking_args)
    else:
        results = model(frame, **inference_args)

    annotated_frame = results[0].plot()

    if frame_count % 10 == 0:
        detected_objects = []
        for box_data in results[0].boxes.data:
            if ENABLE_TRACKING and len(box_data) > 6:
                track_id = int(box_data[4])
                conf = float(box_data[5])
                cls_id = int(box_data[6])
                label = CLASS_NAMES[cls_id]
                detected_objects.append(f"ID:{track_id} {label} ({conf:.2f})")
            elif len(box_data) > 5:
                conf = float(box_data[4])
                cls_id = int(box_data[5])
                label = CLASS_NAMES[cls_id]
                detected_objects.append(f"{label} ({conf:.2f})")


        if detected_objects:
            print(f"[🔍 Frame {frame_count}] Detected: {', '.join(detected_objects)}")
        else:
            print(f"[🔍 Frame {frame_count}] Nothing detected.")

    mode_text = "MODE: PERSON ONLY" if SHOW_ONLY_PERSON else "MODE: ALL OBJECTS (80 COCO)"
    tracking_text = f"TRACKING: {'ON' if ENABLE_TRACKING else 'OFF'}"
    model_info_text = f"Model: {MODEL_NAME} | Conf: {CONF_THRESHOLD:.2f} | FPS: {int(fps)}"

    cv2.putText(annotated_frame, mode_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(annotated_frame, tracking_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(annotated_frame, model_info_text, (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)

    cv2.imshow("Omni-Detector v1.2 — Press 'q' to Quit", annotated_frame)

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

    elif key == ord('k'):
        ENABLE_TRACKING = not ENABLE_TRACKING
        status = "ENABLED" if ENABLE_TRACKING else "DISABLED"
        print(f"[➡️] Object Tracking {status}!")

cap.release()
cv2.destroyAllWindows()
print("[✅] Omni-Detector terminated. Goodbye!")