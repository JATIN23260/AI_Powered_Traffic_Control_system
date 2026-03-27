import cv2
import torch
from ultralytics import YOLO

# 1. Setup Model & GPU
model = YOLO('./models/yolov8n.pt') # 'l' (large) - higher accuracy model
if torch.cuda.is_available():
    model.to('cuda')

cap = cv2.VideoCapture('./videos/video1.mp4')

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

LINE_LEFT   = int(width * 0.01)
LINE_RIGHT  = int(width * 0.99)
LINE_TOP    = int(height * 0.01)
LINE_BOTTOM = int(height * 0.99)

track_history = {}
frame_count = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_count += 1
    if frame_count % 3 != 0: continue

    results = model.track(frame, persist=True, conf=0.2, imgsz=max(width, height), classes=[1,2,3,5,7])

    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if track_id in track_history:
                px, py = track_history[track_id] # Previous X, Previous Y

                # WEST (Left Line)
                if px < LINE_LEFT and cx >= LINE_LEFT: print(f"Vehicle {track_id} ENTERED from West")
                elif px > LINE_LEFT and cx <= LINE_LEFT: print(f"Vehicle {track_id} EXITED to West")

                # EAST (Right Line)
                if px > LINE_RIGHT and cx <= LINE_RIGHT: print(f"Vehicle {track_id} ENTERED from East")
                elif px < LINE_RIGHT and cx >= LINE_RIGHT: print(f"Vehicle {track_id} EXITED to East")

                # NORTH (Top Line)
                if py < LINE_TOP and cy >= LINE_TOP: print(f"Vehicle {track_id} ENTERED from North")
                elif py > LINE_TOP and cy <= LINE_TOP: print(f"Vehicle {track_id} EXITED to North")

                # SOUTH (Bottom Line)
                if py > LINE_BOTTOM and cy <= LINE_BOTTOM: print(f"Vehicle {track_id} ENTERED from South")
                elif py < LINE_BOTTOM and cy >= LINE_BOTTOM: print(f"Vehicle {track_id} EXITED to South")

            track_history[track_id] = (cx, cy)
            
            # Draw tracking visuals
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    cv2.line(frame, (LINE_LEFT, 0), (LINE_LEFT, height), (0, 255, 0), 2)   # West
    cv2.line(frame, (LINE_RIGHT, 0), (LINE_RIGHT, height), (0, 255, 0), 2) # East
    cv2.line(frame, (0, LINE_TOP), (width, LINE_TOP), (0, 255, 255), 2)    # North
    cv2.line(frame, (0, LINE_BOTTOM), (width, LINE_BOTTOM), (0, 255, 255), 2) # South

    cv2.imshow("4-Way AI Sensor Grid", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()