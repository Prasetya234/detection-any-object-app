import math

import cv2
import cvzone
import numpy as np
from ultralytics import YOLO

from model.yolo import classNames
from utils.sort import Sort

model = YOLO("datasets/yolov8l.pt")


class Example1:
    def __init__(self, still_running=False):
        self.category = ""
        self.source = ""
        self.label = ""
        self.count = 0
        self.still_running = still_running
        self.tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    def image_sys(self):
        frame = cv2.imread(self.source)
        frame = cv2.resize(frame, (1200, 720))

        models = model(source=frame, stream=True)

        for r in models:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                # cls = int(box.cls[0])
                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255,0,0))
                # cv2.circle(frame, (cx, cy), 5, (255,0,0), cv2.FILLED)
                self.count += 1

            cvzone.putTextRect(frame, f'Count: {self.count}', (10, 670), scale=1, thickness=2, offset=10)
        cvzone.putTextRect(frame, f'Object name: {self.label}', (10, 700), scale=1, thickness=2,offset=10)

        cv2.imshow("Result ", frame)
        # cv2.imwrite('results/test1.png', frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def video_sys(self):
        cap = cv2.VideoCapture(self.source)
        cap.set(3, 1200)
        cap.set(4, 720)

        while cap.isOpened():
            ret, frame = cap.read()

            if ret is False:
                print("Image can't load")
                break

            tracker = self.tracker

            models = model(source=frame, stream=True)
            detections = np.empty((0, 5))
            for r in models:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1

                    # conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])

                    # cx, cy = x1 + w // 2, y1 + h // 2
                    currentArray = np.array([x1, y1, x2, y2, cls])
                    detections = np.vstack((detections, currentArray))

            resultsTracker = tracker.update(detections)

            for result in resultsTracker:
                x1, y1, x2, y2, cls = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cx, cy = x1 + w // 2, y1 + h // 2

                cvzone.cornerRect(frame, (x1, y1, w, h), l=9, rt=2, colorR=(255,255,0))
                # cv2.circle(frame, (cx, cy), 5, (255,255,0), cv2.FILLED)
                self.count += 1


            cvzone.putTextRect(frame, f'Count: {self.count}', (max(35, 20), max(0, 650)), scale=2, thickness=3,
                               offset=10)
            cvzone.putTextRect(frame, f'Category: {self.label}', (max(35, 20), max(0, 700)), scale=2, thickness=3,
                               offset=10)
            cv2.imshow("Result ", frame)
            # cv2.imwrite('images/test01.png',frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()

    def run_main(self, type, video):
        if self.still_running == False:
            print("Program Can't running")
            return

        if type == "video":
            self.source = "assets/video/" + video
            self.video_sys()
        else:
            self.source = "assets/image/" + video
            self.image_sys()
