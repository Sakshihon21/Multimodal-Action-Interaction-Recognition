"""
Face Detection and Emotion Recognition Module
Uses OpenCV for face detection and can integrate with emotion recognition models
"""

import cv2
import numpy as np
import os

class FaceEmotionDetector:
    def __init__(self):
        self.enabled = True
        self.face_cascade = None
        self.emotion_model = None
        
        try:
            # Load OpenCV face cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
                print("Face cascade loaded successfully")
            else:
                # Try alternative path
                alt_path = 'haarcascade_frontalface_default.xml'
                if os.path.exists(alt_path):
                    self.face_cascade = cv2.CascadeClassifier(alt_path)
                else:
                    print("Face cascade not found")
                    self.enabled = False
        except Exception as e:
            print(f"Face detection initialization failed: {e}")
            self.enabled = False
        
        # Emotion labels (for when model is available)
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        
        # Try to load emotion recognition model
        self.load_emotion_model()
    
    def load_emotion_model(self):
        """Try to load emotion recognition model"""
        try:
            # You can integrate with FER (Facial Expression Recognition) library
            # or use a pre-trained model like fer2013
            try:
                from fer import FER
                self.emotion_detector = FER(mtcnn=True)
                self.fer_available = True
                print("FER emotion detector loaded")
            except ImportError:
                self.fer_available = False
                # Try tensorflow/keras model
                try:
                    import tensorflow as tf
                    # Load your trained emotion model here
                    # self.emotion_model = tf.keras.models.load_model('emotion_model.h5')
                    self.tf_available = False
                except:
                    self.tf_available = False
        except Exception as e:
            print(f"Emotion model loading failed: {e}")
            self.fer_available = False
            self.tf_available = False
    
    def detect_face_emotion(self, img):
        """Detect faces and emotions in the image"""
        if not self.enabled or self.face_cascade is None:
            return img, None
        
        emotion = None
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        # Process each face
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Extract face ROI
            face_roi = gray[y:y+h, x:x+w]
            
            # Try emotion detection
            if self.fer_available:
                try:
                    emotions = self.emotion_detector.detect_emotions(img)
                    if emotions:
                        emotion_data = emotions[0]['emotions']
                        emotion = max(emotion_data, key=emotion_data.get)
                        confidence = emotion_data[emotion]
                        
                        # Draw emotion label
                        cv2.putText(img, f"{emotion}: {confidence:.2f}", 
                                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.7, (255, 255, 0), 2)
                except:
                    pass
            else:
                # Basic emotion estimation using facial features
                emotion = self.estimate_emotion_basic(face_roi)
                if emotion:
                    cv2.putText(img, f"Emotion: {emotion}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, (255, 255, 0), 2)
        
        return img, emotion
    
    def estimate_emotion_basic(self, face_roi):
        """Basic emotion estimation using simple heuristics"""
        # This is a simplified version - in production, use a trained model
        # For now, we'll use some basic features
        
        if face_roi.size == 0:
            return None
        
        # Calculate some basic features
        brightness = np.mean(face_roi)
        
        # Simple heuristic (not very accurate, but demonstrates the concept)
        if brightness > 120:
            return "Happy"
        elif brightness < 80:
            return "Neutral"
        else:
            return "Neutral"
    
    def get_emotion_confidence(self, emotion_data):
        """Get confidence score for detected emotion"""
        if not emotion_data:
            return None
        return max(emotion_data.values())

