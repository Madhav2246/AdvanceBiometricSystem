import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class EmotionAnalyzer:
    def __init__(self, model_path="emotion_model.h5"):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path if os.path.isabs(model_path) else os.path.join(base_dir, model_path)
        self.emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        self.model = None
        self.confidence_threshold = 0.25

        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"Emotion model loaded: {self.model_path}")
        else:
            print(f"Warning: Emotion model not found: {self.model_path}. Emotion will be disabled.")

    def detect_emotion(self, face_image, return_confidence=False):
        try:
            if self.model is None:
                return ("Unknown", 0.0) if return_confidence else "Unknown"
            if face_image is None or face_image.size == 0:
                return ("Unknown", 0.0) if return_confidence else "Unknown"

            # Normalize and resize to match training (48x48 grayscale)
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            resized = cv2.resize(gray, (48, 48))
            normalized = resized.astype("float32") / 255.0
            input_tensor = normalized.reshape(1, 48, 48, 1)

            preds = self.model.predict(input_tensor, verbose=0)[0]
            idx = int(np.argmax(preds))
            confidence = float(preds[idx])
            if idx >= len(self.emotions):
                return ("Unknown", confidence) if return_confidence else "Unknown"
            # Lower threshold to avoid always-unknown in real lighting
            if confidence < self.confidence_threshold:
                return ("Unknown", confidence) if return_confidence else "Unknown"
            result = self.emotions[idx]
            return (result, confidence) if return_confidence else result
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return ("Unknown", 0.0) if return_confidence else "Unknown"
