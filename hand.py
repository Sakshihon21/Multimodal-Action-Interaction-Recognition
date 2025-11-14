"""
Advanced Computer Vision Application: Multi-Modal Gesture Control System
Features:
- Multi-hand tracking and gesture recognition
- Object detection for scene understanding
- Face detection and emotion recognition
- Gesture sequence recognition
- Performance analytics
- Real-time visualization
"""

import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import pyautogui
import time
from collections import deque
import os

# Import custom modules (local files in this folder)
try:
    from object_detector import ObjectDetector
    from face_emotion_detector import FaceEmotionDetector
    from gesture_sequence import GestureSequenceRecognizer
    from performance_analytics import PerformanceAnalytics
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some modules not available ({e}). Running in basic mode.")
    MODULES_AVAILABLE = False

class AdvancedGestureControl:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=2, detectionCon=0.8)

        
        self.screen_width, self.screen_height = pyautogui.size()
        self.draw_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.prev_x, self.prev_y = 0, 0
        self.draw_mode = False
        self.control_enabled = True
        
        # Gesture sequence tracking
        self.gesture_history = deque(maxlen=10)
        self.last_gesture_time = time.time()
        
        # Initialize advanced modules
        if MODULES_AVAILABLE:
            self.object_detector = ObjectDetector()
            self.face_emotion = FaceEmotionDetector()
            self.gesture_sequence = GestureSequenceRecognizer()
            self.performance = PerformanceAnalytics()
        
        # Color modes
        self.colors = {
            'draw': (0, 255, 255),  # Cyan
            'select': (255, 0, 0),  # Blue
            'click': (0, 255, 0),   # Green
            'warning': (0, 0, 255)  # Red
        }
        
        # Drawing settings
        self.brush_size = 5
        self.current_color = self.colors['draw']
        
    def recognize_gesture(self, fingers):
        """Map finger configuration to gesture name"""
        gestures = {
            (0, 0, 0, 0, 0): "FIST",
            (1, 1, 1, 1, 1): "OPEN_PALM",
            (0, 1, 0, 0, 0): "POINT",
            (0, 1, 1, 0, 0): "VICTORY",
            (1, 0, 0, 0, 0): "THUMB_UP",
            (0, 0, 0, 0, 1): "PINKY",
            (1, 1, 0, 0, 0): "OK",
            (0, 1, 1, 1, 0): "FOUR",
            (1, 1, 1, 0, 0): "THREE",
        }
        finger_tuple = tuple(fingers)
        return gestures.get(finger_tuple, "UNKNOWN")
    
    def process_hand_gestures(self, hands, img):
        """Process multi-hand gestures with advanced recognition"""
        if not hands:
            return img
        
        # Process multiple hands
        for idx, hand in enumerate(hands):
            lm_list = hand['lmList']
            fingers = self.detector.fingersUp(hand)
            hand_type = hand['type']  # 'Left' or 'Right'
            
            x, y = lm_list[8][0], lm_list[8][1]  # Index fingertip
            cx, cy = x, y
            
            gesture_name = self.recognize_gesture(fingers)
            self.gesture_history.append(gesture_name)
            
            # Display gesture info
            cv2.putText(img, f"{hand_type} Hand: {gesture_name}", 
                       (10, 30 + idx * 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.7, (255, 255, 255), 2)
            
            # === Advanced Gesture Controls ===
            
            # Smart Toggle
            if gesture_name == "FIST":
                self.control_enabled = False
                cv2.putText(img, "Control OFF", (100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['warning'], 3)
            
            elif gesture_name == "OPEN_PALM":
                self.control_enabled = True
                cv2.putText(img, "Control ON", (100, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['click'], 3)
            
            if self.control_enabled:
                # Drawing Mode: Victory sign
                if gesture_name == "VICTORY":
                    self.draw_mode = True
                    cv2.putText(img, "Drawing Mode", (100, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                    if self.prev_x == 0 and self.prev_y == 0:
                        self.prev_x, self.prev_y = cx, cy
                    cv2.line(self.draw_canvas, (self.prev_x, self.prev_y), 
                            (cx, cy), self.current_color, self.brush_size)
                    self.prev_x, self.prev_y = cx, cy
                
                # Mouse Move: Point gesture
                elif gesture_name == "POINT":
                    self.draw_mode = False
                    self.prev_x, self.prev_y = 0, 0
                    screen_x = np.interp(cx, [0, 1280], [0, self.screen_width])
                    screen_y = np.interp(cy, [0, 720], [0, self.screen_height])
                    pyautogui.moveTo(screen_x, screen_y)
                
                # Click: OK gesture
                elif gesture_name == "OK":
                    x1, y1 = lm_list[4][0], lm_list[4][1]  # Thumb tip
                    distance = np.hypot(x1 - cx, y1 - cy)
                    if distance < 30:
                        pyautogui.click()
                        cv2.circle(img, (cx, cy), 15, self.colors['click'], cv2.FILLED)
                        cv2.putText(img, "Click!", (100, 200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.colors['click'], 3)
                
                # Right Click: Thumb Up
                elif gesture_name == "THUMB_UP":
                    pyautogui.rightClick()
                    cv2.putText(img, "Right Click!", (100, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 3)
                
                # Change brush color: Three fingers
                elif gesture_name == "THREE" and len(hands) == 1:
                    color_index = int(time.time()) % len(list(self.colors.values()))
                    self.current_color = list(self.colors.values())[color_index]
                    cv2.putText(img, "Color Changed!", (100, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.current_color, 2)
                
                # Clear canvas: Four fingers
                elif gesture_name == "FOUR":
                    self.draw_canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(img, "Canvas Cleared!", (100, 250), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Draw hand landmarks
            for lm in lm_list:
                cv2.circle(img, (lm[0], lm[1]), 5, (255, 0, 255), cv2.FILLED)
        
        return img
    
    def process_advanced_features(self, img):
        """Process object detection, face/emotion, and gesture sequences"""
        if not MODULES_AVAILABLE:
            return img
        
        # Object Detection
        img = self.object_detector.detect_objects(img)
        
        # Face and Emotion Detection
        img, emotion = self.face_emotion.detect_face_emotion(img)
        if emotion:
            cv2.putText(img, f"Emotion: {emotion}", (10, img.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Gesture Sequence Recognition
        if len(self.gesture_history) >= 3:
            sequence_result = self.gesture_sequence.recognize_sequence(list(self.gesture_history))
            if sequence_result:
                cv2.putText(img, f"Sequence: {sequence_result}", (10, img.shape[0] - 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return img
    
    def draw_ui_overlay(self, img):
        """Draw UI overlay with information"""
        # Performance metrics
        if MODULES_AVAILABLE:
            fps = self.performance.get_fps()
            latency = self.performance.get_latency()
            cv2.putText(img, f"FPS: {fps:.1f}", (img.shape[1] - 150, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img, f"Latency: {latency:.1f}ms", (img.shape[1] - 200, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Gesture legend
        legend_y = 100
        cv2.putText(img, "Gestures:", (img.shape[1] - 200, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        gestures_legend = [
            "Fist: Disable",
            "Open: Enable",
            "Point: Move",
            "V: Draw",
            "OK: Click",
            "Thumb: R-Click"
        ]
        for i, text in enumerate(gestures_legend):
            cv2.putText(img, text, (img.shape[1] - 200, legend_y + 25 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return img
    
    def run(self):
        """Main application loop"""
        print("Starting Advanced Computer Vision Gesture Control System...")
        print("Press ESC to exit")
        
        while True:
            if MODULES_AVAILABLE:
                self.performance.start_frame()
            
            success, img = self.cap.read()
            if not success:
                break
            
            img = cv2.resize(img, (1280, 720))
            img = cv2.flip(img, 1)
            
            # Hand detection
            hands, img = self.detector.findHands(img, draw=True)
            
            # Process gestures
            img = self.process_hand_gestures(hands, img)
            
            # Advanced features
            img = self.process_advanced_features(img)
            
            # Combine drawing canvas
            gray_canvas = cv2.cvtColor(self.draw_canvas, cv2.COLOR_BGR2GRAY)
            _, inv = cv2.threshold(gray_canvas, 50, 255, cv2.THRESH_BINARY_INV)
            inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
            img = cv2.bitwise_and(img, inv)
            img = cv2.bitwise_or(img, self.draw_canvas)
            
            # UI overlay
            img = self.draw_ui_overlay(img)
            
            # Display
            cv2.imshow("Advanced CV: Gesture Control System", img)
            
            if MODULES_AVAILABLE:
                self.performance.end_frame()
            
            if cv2.waitKey(1) & 0xFF == 27:  # ESC key
                break
        
        self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        if MODULES_AVAILABLE:
            self.performance.print_summary()

if __name__ == "__main__":
    app = AdvancedGestureControl()
    app.run()
