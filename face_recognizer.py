import os
import json
import numpy as np
import cv2
import dlib

class FaceRecognizer:
    def __init__(self, storage_path="database.json", recognition_threshold=0.6, max_encodings_per_person=10):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.storage_path = storage_path if os.path.isabs(storage_path) else os.path.join(base_dir, storage_path)
        self.recognition_threshold = recognition_threshold
        self.max_encodings_per_person = max_encodings_per_person
        self.embeddings = {}

        self.detector = dlib.get_frontal_face_detector()

        self.shape_predictor_path = os.path.join(base_dir, "shape_predictor_68_face_landmarks.dat")
        self.rec_model_path = os.path.join(base_dir, "dlib_face_recognition_resnet_model_v1.dat")

        self.shape_predictor = None
        self.rec_model = None

        if os.path.exists(self.shape_predictor_path):
            self.shape_predictor = dlib.shape_predictor(self.shape_predictor_path)
        else:
            print(f"Warning: Missing {self.shape_predictor_path}. Recognition will be disabled.")

        if os.path.exists(self.rec_model_path):
            self.rec_model = dlib.face_recognition_model_v1(self.rec_model_path)
        else:
            print(f"Warning: Missing {self.rec_model_path}. Recognition will be disabled.")

        self.load_encodings()

    def load_encodings(self):
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.embeddings = {name.lower(): [np.array(encoding, dtype=np.float32) for encoding in encodings]
                                       for name, encodings in data.items()}
                print(f"Loaded encodings for {len(self.embeddings)} users from {self.storage_path}")
            except Exception as e:
                print(f"Error loading encodings: {e}")
                self.embeddings = {}
        else:
            print(f"No existing encodings found at {self.storage_path}. Starting fresh.")

    def save_encodings(self):
        try:
            data = {name: [encoding.tolist() for encoding in encodings]
                    for name, encodings in self.embeddings.items()}
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved encodings for {len(self.embeddings)} users to {self.storage_path}")
        except Exception as e:
            print(f"Error saving encodings: {e}")

    def list_users(self):
        return [{"name": name, "samples": len(encs)} for name, encs in sorted(self.embeddings.items())]

    def delete_user(self, user_id):
        user_id = user_id.lower()
        if user_id in self.embeddings:
            self.embeddings.pop(user_id, None)
            self.save_encodings()
            return True
        return False

    def _embed(self, image):
        if self.shape_predictor is None or self.rec_model is None:
            return None

        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image

        # Downscale for speed but keep enough detail
        h, w = rgb.shape[:2]
        if w > 900:
            scale = 900 / w
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))

        dets = self.detector(rgb, 1)
        if len(dets) == 0:
            dets = self.detector(rgb, 2)
        if len(dets) == 0:
            return None

        # Use the largest face
        det = max(dets, key=lambda r: r.width() * r.height())
        if det.width() < 60 or det.height() < 60:
            return None
        shape = self.shape_predictor(rgb, det)
        face_chip = dlib.get_face_chip(rgb, shape, size=150)
        descriptor = self.rec_model.compute_face_descriptor(face_chip)
        return np.array(descriptor, dtype=np.float32)

    def enroll(self, get_next_frame, user_id, progress_cb=None):
        user_id = user_id.lower()
        if self.shape_predictor is None or self.rec_model is None:
            if progress_cb:
                progress_cb("error", "Required dlib models are missing.")
            return False

        encodings_list = []
        angles = ["front", "left", "right", "up", "down"]
        for i, angle in enumerate(angles):
            if progress_cb:
                progress_cb("prompt", f"Show {angle} angle ({i+1}/{len(angles)})")
            frame = get_next_frame()
            if frame is None:
                if progress_cb:
                    progress_cb("warn", "Failed to capture frame for enrollment.")
                continue

            attempts = 0
            embedding = None
            while attempts < 5 and embedding is None:
                embedding = self._embed(frame)
                if embedding is None:
                    attempts += 1
                    frame = get_next_frame()
                    if frame is None:
                        break

            if embedding is None:
                if progress_cb:
                    progress_cb("warn", f"No face found for {angle} angle.")
                continue
            encodings_list.append(embedding)

        if len(encodings_list) < 2:
            if progress_cb:
                progress_cb("error", "Enrollment failed: Not enough valid samples.")
            return False

        if user_id not in self.embeddings:
            self.embeddings[user_id] = []

        self.embeddings[user_id].extend(encodings_list)
        if len(self.embeddings[user_id]) > self.max_encodings_per_person:
            self.embeddings[user_id] = self.embeddings[user_id][-self.max_encodings_per_person:]
        if progress_cb:
            progress_cb("done", f"Enrolled {user_id} with {len(self.embeddings[user_id])} samples.")
        self.save_encodings()
        return True

    def _euclidean_distance(self, a, b):
        return float(np.linalg.norm(a - b))

    def recognize(self, image):
        embedding = self._embed(image)
        if embedding is None:
            return "Unknown"

        best_match = "Unknown"
        best_distance = float('inf')
        for name, stored_encodings in self.embeddings.items():
            for enc in stored_encodings:
                distance = self._euclidean_distance(enc, embedding)
                if distance < best_distance:
                    best_distance = distance
                    best_match = name if distance < self.recognition_threshold else "Unknown"

        return best_match
