"""
Object Detection Module using YOLOv8
Detects objects in the scene for enhanced context understanding
"""

import cv2
import numpy as np
import os

class ObjectDetector:
    def __init__(self):
        self.enabled = True
        try:
            # Try to use YOLO if available
            try:
                from ultralytics import YOLO
                self.model = YOLO('yolov8n.pt')  # Nano model for speed
                self.yolo_available = True
                print("YOLO model loaded successfully")
            except ImportError:
                print("YOLO not available. Using basic detection.")
                self.yolo_available = False
                # Fallback to OpenCV's DNN
                self.init_opencv_dnn()
        except Exception as e:
            print(f"Object detection initialization failed: {e}")
            self.enabled = False
            self.yolo_available = False
    
    def init_opencv_dnn(self):
        """Initialize OpenCV DNN as fallback"""
        try:
            # You can download these files from OpenCV repository
            config_path = "yolo/yolov3.cfg"
            weights_path = "yolo/yolov3.weights"
            
            if os.path.exists(config_path) and os.path.exists(weights_path):
                self.net = cv2.dnn.readNet(weights_path, config_path)
                self.opencv_dnn_available = True
            else:
                self.opencv_dnn_available = False
        except:
            self.opencv_dnn_available = False
    
    def detect_objects_yolo(self, img):
        """Detect objects using YOLOv8"""
        if not self.yolo_available:
            return img
        
        try:
            results = self.model(img, verbose=False)
            
            # Draw detections
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    class_name = self.model.names[class_id]
                    
                    if confidence > 0.5:  # Confidence threshold
                        # Draw bounding box
                        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), 
                                     (0, 255, 0), 2)
                        # Draw label
                        label = f"{class_name}: {confidence:.2f}"
                        cv2.putText(img, label, (int(x1), int(y1) - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return img
    
    def detect_objects_opencv(self, img):
        """Basic object detection using OpenCV"""
        # Simple color-based detection as fallback
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Detect common objects by color (example: detecting red objects)
        lower_red = np.array([0, 120, 70])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Object", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return img
    
    def detect_objects(self, img):
        """Main detection method"""
        if not self.enabled:
            return img
        
        if self.yolo_available:
            return self.detect_objects_yolo(img)
        elif hasattr(self, 'opencv_dnn_available') and self.opencv_dnn_available:
            return self.detect_objects_opencv(img)
        else:
            # Minimal detection - just show a message
            cv2.putText(img, "Object Detection: YOLO not available", 
                       (10, img.shape[0] - 80), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (128, 128, 128), 2)
            return img

