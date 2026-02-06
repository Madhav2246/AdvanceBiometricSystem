import cv2
import numpy as np
import os

class FaceDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cascade_path = os.path.join(base_dir, "haarcascade_frontalface_default.xml")
        if not os.path.exists(cascade_path):
            raise FileNotFoundError(f"Haar cascade file not found: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        self.liveness_state = {}
        self.state_ttl_frames = 30
        self.motion_threshold = 12.0
        self.min_motion_frames = 1

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(40, 40)
        )
        results = []
        for (x, y, w, h) in faces:
            results.append((x, y, x + w, y + h))
        return results

    def validate_face(self, frame, face_box):
        x, y, x1, y1 = face_box
        x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
        x = max(0, x)
        y = max(0, y)
        x1 = min(frame.shape[1], x1)
        y1 = min(frame.shape[0], y1)

        if x1 <= x or y1 <= y:
            return False

        width = x1 - x
        height = y1 - y
        aspect_ratio = width / height if height != 0 else 0
        if not (0.5 <= aspect_ratio <= 2.2):
            return False

        return True

    def _get_state(self, face_id):
        if face_id not in self.liveness_state:
            self.liveness_state[face_id] = {
                "last_frame": None,
                "motion_frames": 0,
                "last_seen": 0
            }
        return self.liveness_state[face_id]

    def _cleanup_state(self, current_frame_idx):
        stale_ids = [k for k, v in self.liveness_state.items()
                     if current_frame_idx - v["last_seen"] > self.state_ttl_frames]
        for k in stale_ids:
            self.liveness_state.pop(k, None)

    def is_alive(self, frame, face_box, face_id=None, frame_idx=0):
        face_key = face_id if face_id is not None else "global"
        state = self._get_state(face_key)
        state["last_seen"] = frame_idx
        self._cleanup_state(frame_idx)

        x, y, x1, y1 = face_box
        x, y, x1, y1 = int(x), int(y), int(x1), int(y1)
        x = max(0, x)
        y = max(0, y)
        x1 = min(frame.shape[1], x1)
        y1 = min(frame.shape[0], y1)
        if x1 <= x or y1 <= y:
            return False, "invalid_box"

        roi = frame[y:y1, x:x1]
        if roi.size == 0:
            return False, "empty_roi"

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        if state["last_frame"] is None:
            state["last_frame"] = gray
            return False, "checking"

        if state["last_frame"].shape != gray.shape:
            state["last_frame"] = gray
            state["motion_frames"] = 0
            return False, "resized"

        diff = cv2.absdiff(state["last_frame"], gray)
        motion = float(np.mean(diff))
        state["last_frame"] = gray

        if motion > self.motion_threshold:
            state["motion_frames"] += 1
        else:
            state["motion_frames"] = max(0, state["motion_frames"] - 1)

        if state["motion_frames"] >= self.min_motion_frames:
            return True, "motion_detected"

        return False, "no_motion"
