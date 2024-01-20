import math
class Tracker:
    def __init__(self):
        self.center_points = {}
        self.id_count = 0
    def update(self, objects_rect):
        objects_bbs_ids = []
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])
                if dist < 35:
                    self.center_points[id] = (cx, cy)
                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center
        self.center_points = new_center_points.copy()
        return objects_bbs_ids




import cv2
import pandas as pd
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
#cap = cv2.VideoCapture(0)  # To access your camera you can chnge it as 1 while using external camera
cap = cv2.VideoCapture('people.mp4') # To use customized video
my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
count = 0
entries = []
exits = []
tracker = Tracker()
cy1 = 205
cy2 = 270
offset = 6
peo_in = {}
peo_out = {}
#while cap.isOpened(): # While using your camera
while True:  # While using custom video  
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 3 != 0:
        continue
    frame = cv2.resize(frame, (1020, 500))
    results = model.predict(frame)
    a = results[0].boxes.data
    px = pd.DataFrame(a).astype("float")
    bbox_list = []
    for index, row in px.iterrows():
        x1, y1, x2, y2, _, d = row
        c = class_list[int(d)]
        if 'person' in c:
            bbox_list.append([x1, y1, x2, y2])
    bbox_id = tracker.update(bbox_list)
    
    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)
        
        if (cy1 - offset) < cy < (cy1 + offset):
            peo_in[id] = cy
            if id not in exits and id not in entries:
                entries.append(id)
                #cv2.rectangle(frame,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,255),2)
                #cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                #cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
                
        if (cy2 - offset) < cy < (cy2 + offset):
            peo_out[id] = cy
            if id not in exits and id not in entries:
                exits.append(id)
                #cv2.rectangle(frame,(int(x3),int(y3)),(int(x4),int(y4)),(255,0,255),2)
                #cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                #cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)

    cv2.line(frame, (0, cy1), (1019, cy1), (255, 255, 255), 1)
    cv2.line(frame, (0, cy2), (1019, cy2), (255, 255, 255), 1)
    b = len(entries)
    c = len(exits)
    cv2.putText(frame, f'Entries: {b}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f'Exits: {c}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q' or 'Q'):
        break

cap.release()
cv2.destroyAllWindows()
