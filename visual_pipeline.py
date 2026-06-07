import cv2
from deepface import DeepFace
import threading

class VisualPipeline:
    EMOTION_WEIGHTS = {
        'happy': 1.0,
        'surprise': 0.5,
        'neutral': 0.0,
        'fear': -0.3,
        'disgust': -0.5,
        'sad': -0.7,
        'angry': -1.0
    }

    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.current_frame = None
        self.current_emotions = None
        self.dominant_emotion = None
        self.face_detected = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def start_camera(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            return False
        return True

    def get_frame(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
            
        frame = cv2.flip(frame, 1)
        self.current_frame = frame
        return frame

    def analyze_emotion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            self.face_detected = False
            self.current_emotions = None
            self.dominant_emotion = None
            return None

        self.face_detected = True
        
        try:
            results = DeepFace.analyze(
                frame, 
                actions=['emotion'], 
                enforce_detection=False,
                silent=True
            )
            
            if isinstance(results, list):
                result = results[0]
            else:
                result = results

            emotions_raw = result['emotion']
            
            total = sum(emotions_raw.values())
            self.current_emotions = {k: v/total for k, v in emotions_raw.items()}
            self.dominant_emotion = result['dominant_emotion']
            
            return self.current_emotions
            
        except Exception:
            self.current_emotions = None
            self.dominant_emotion = None
            return None

    def get_sentiment_score(self):
        if not self.current_emotions or not self.face_detected:
            return 0.0, 0.0
            
        score = 0.0
        for emotion, prob in self.current_emotions.items():
            weight = self.EMOTION_WEIGHTS.get(emotion, 0.0)
            score += weight * prob
            
        confidence = max(self.current_emotions.values())
        return score, confidence

    def release(self):
        if self.cap is not None:
            self.cap.release()
