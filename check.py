import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, weights_path="best.pt"):
        print(f"Loading YOLOv8 model from: {weights_path}")
        # 1. Khởi tạo mô hình YOLOv8
        self.model = YOLO(weights_path)
        
        # Lấy danh sách class trực tiếp từ file .pt
        self.classes = self.model.names 
        
        # Cấu hình vạch đếm và bộ đếm
        self.line_y = 550 
        self.already_counted = set() 
        self.target_classes = ["car", "truck", "motorbike", 'bus', 'pickup-van', 'microbus']
        self.counters = {cls: 0 for cls in self.target_classes}

    def run(self, video_path):
        cap = cv2.VideoCapture(video_path)
        print("Opening:", video_path)

        if not cap.isOpened():
            print("Cannot open video!")
            return
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # 2. Sử dụng tính năng Track tích hợp của YOLOv8 (ByteTrack)
            # persist=True giúp giữ ID của đối tượng qua từng frame
            results = self.model.track(frame, persist=True, conf=0.5, iou=0.4, verbose=False)

            if results[0].boxes.id is not None:
                # Lấy tọa độ (xyxy), IDs, và Class Indices
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                ids = results[0].boxes.id.cpu().numpy().astype(int)
                clss = results[0].boxes.cls.cpu().numpy().astype(int)

                for box, obj_id, cls_idx in zip(boxes, ids, clss):
                    label = self.classes[cls_idx]
                    
                    # Chỉ xử lý các loại xe trong danh sách target
                    if label in self.target_classes:
                        x1, y1, x2, y2 = box
                        # Tính tâm đối tượng
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        # Vẽ khung và ID
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID:{obj_id} {label}", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

                        # 3. Logic đếm xe khi đi qua vạch (cy > line_y)
                        if cy > self.line_y and obj_id not in self.already_counted:
                            self.counters[label] += 1
                            self.already_counted.add(obj_id)

            # Vẽ vạch đếm (màu vàng)
            cv2.line(frame, (0, self.line_y), (frame.shape[1], self.line_y), (0, 255, 255), 3)
            
            # Hiển thị bảng đếm lên góc màn hình
            for i, (name, count) in enumerate(self.counters.items()):
                cv2.putText(frame, f"{name.upper()}: {count}", (20, 150 + i*35), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("YOLOv8 Vehicle Counting", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'): # Nhấn 'q' để thoát
                break

        cap.release()
        cv2.destroyAllWindows()

# Khởi chạy thực tế
if __name__ == "__main__":
    detector = ObjectDetection(weights_path="best.pt")
    detector.run("demo.mp4")
