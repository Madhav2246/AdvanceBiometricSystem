import os
import time
import threading
import queue
import traceback
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request

from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from emotion_analyzer import EmotionAnalyzer

app = Flask(__name__)

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server.log")

def log_line(msg):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(f"{now_iso()} {msg}\n")
    except Exception:
        pass

WINDOW_TITLE = "Advance Face Biometric System"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EVENT_PATH = os.path.join(BASE_DIR, "events.jsonl")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
QUOTE_CACHE = {"emotion": None, "quote": None, "ts": 0}

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def log_event(payload):
    try:
        payload = dict(payload)
        payload["timestamp"] = now_iso()
        with open(EVENT_PATH, "a", encoding="utf-8") as f:
            f.write(json_dumps(payload) + "\n")
    except Exception as e:
        print(f"Event log error: {e}")

def json_dumps(payload):
    import json
    return json.dumps(payload, ensure_ascii=True)

def generate_quote(emotion):
    if not GEMINI_API_KEY:
        return "Set GEMINI_API_KEY to enable quotes."
    # Simple cache to avoid excessive calls
    now = time.time()
    if QUOTE_CACHE["emotion"] == emotion and now - QUOTE_CACHE["ts"] < 30:
        return QUOTE_CACHE["quote"]

    prompt = (
        "Generate one short, uplifting, professional 'good vibe' quote "
        f"tailored to someone feeling {emotion}. Keep it under 20 words."
    )
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    body = {
        "contents": [
            {"parts": [{"text": prompt}]}
        ]
    }
    try:
        resp = requests.post(f"{url}?key={GEMINI_API_KEY}", headers=headers, json=body, timeout=10)
        if resp.status_code != 200:
            return "Quote service unavailable."
        data = resp.json()
        candidates = data.get("candidates", [])
        if not candidates:
            return "Stay positive. You're doing great."
        text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "").strip()
        if not text:
            return "Keep going. Small steps create big change."
        QUOTE_CACHE["emotion"] = emotion
        QUOTE_CACHE["quote"] = text
        QUOTE_CACHE["ts"] = now
        return text
    except Exception as e:
        log_line(f"Quote error: {e}")
        return "Keep going. You've got this."

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area != 0 else 0

def blur_region(frame, box, ksize=31):
    x, y, x1, y1 = box
    x, y, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 <= x or y1 <= y:
        return
    roi = frame[y:y1, x:x1]
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    frame[y:y1, x:x1] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

class CameraStream:
    def __init__(self, cam_index=0):
        self.cam_index = cam_index
        self.cap = None
        self.lock = threading.Lock()
        self.latest_frame = None
        self.frame_id = 0
        self.running = False
        self.thread = None

    def start(self):
        if self.running:
            return
        self.running = True
        self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def switch(self, cam_index):
        with self.lock:
            self.cam_index = int(cam_index)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _loop(self):
        while self.running:
            try:
                if self.cap is None or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
                    time.sleep(0.2)
                    continue
                ret, frame = self.cap.read()
                if not ret:
                    if self.cap is not None:
                        self.cap.release()
                    time.sleep(0.2)
                    continue
                with self.lock:
                    self.latest_frame = frame
                    self.frame_id += 1
                time.sleep(0.001)
            except Exception as e:
                log_line(f"Camera loop error: {e}")
                log_line(traceback.format_exc())
                time.sleep(0.5)

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None, self.frame_id
            return self.latest_frame.copy(), self.frame_id

class FacePipeline:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.emotion_analyzer = EmotionAnalyzer()

        self.frame_count = 0
        self.detection_interval = 2
        self.recognition_interval = 6
        self.emotion_update_interval = 6
        self.max_missed_frames = 20
        self.iou_threshold = 0.4
        self.padding = 40
        self.min_size = 50

        self.face_data = {}
        self.next_face_id = 0
        self.last_faces = []

        self.consent_enabled = False
        self.blur_unknown = True

        self.last_fps = 0.0
        self.last_status = {"faces": []}
        self.last_annotated = None
        self.last_process_time = 0.0
        self.process_lock = threading.Lock()
        self.metrics_lock = threading.Lock()
        self.recognition_events = []
        self.history = []
        self.enroll_lock = threading.Lock()
        self.enroll_state = {"status": "idle", "message": ""}
        self.unknown_first_seen = {}
        self.unknown_labels = {}
        self.unknown_counter = 1
        self.emotion_window = []
        self.emotion_window_size = 5
        self.emotion_confidence = 0.0

    def set_consent(self, value):
        self.consent_enabled = bool(value)
        log_event({"event": "consent_toggled", "consent": self.consent_enabled})

    def set_blur(self, value):
        self.blur_unknown = bool(value)

    def process_frame(self, frame):
        start_time = time.time()
        # Downscale for smoother processing
        h, w = frame.shape[:2]
        if w > 720:
            scale = 720 / w
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

        if self.frame_count % self.detection_interval == 0:
            faces = self.detector.detect_faces(frame)
        else:
            faces = self.last_faces

        new_face_data = {}
        for face in faces:
            x, y, x1, y1 = face
            best_iou = 0
            best_face_id = None

            for face_id, data in self.face_data.items():
                iou = calculate_iou(face, data["box"])
                if iou > best_iou and iou > self.iou_threshold:
                    best_iou = iou
                    best_face_id = face_id

            if best_face_id is not None:
                new_face_data[best_face_id] = self.face_data[best_face_id]
                new_face_data[best_face_id]["box"] = (x, y, x1, y1)
            else:
                new_face_data[self.next_face_id] = {
                    "box": (x, y, x1, y1),
                    "user_id": "Unknown",
                    "emotion": "Unknown",
                    "recognition_queue": queue.Queue(maxsize=1),
                    "emotion_queue": queue.Queue(maxsize=1),
                    "last_seen": self.frame_count,
                    "last_recognized_frame": -self.recognition_interval,
                    "recognition_inflight": False,
                    "emotion_inflight": False,
                    "liveness_ok": False,
                    "liveness_reason": "checking",
                    "last_logged_user_id": None
                }
                self.next_face_id += 1

        self.face_data = {k: v for k, v in new_face_data.items() if self.frame_count - v["last_seen"] < self.max_missed_frames}
        self.last_faces = [data["box"] for data in self.face_data.values()]

        status_faces = []
        largest_face_area = 0
        frame_area = frame.shape[0] * frame.shape[1]
        for face_id, data in self.face_data.items():
            x, y, x1, y1 = data["box"]
            data["last_seen"] = self.frame_count

            if not self.detector.validate_face(frame, (x, y, x1, y1)):
                continue
            area = max(0, (x1 - x) * (y1 - y))
            if area > largest_face_area:
                largest_face_area = area

            x_roi = max(0, x - self.padding)
            y_roi = max(0, y - self.padding)
            x1_roi = min(frame.shape[1], x1 + self.padding)
            y1_roi = min(frame.shape[0], y1 + self.padding)

            face_roi = frame[y_roi:y1_roi, x_roi:x1_roi]
            if face_roi.size == 0 or face_roi.shape[0] < self.min_size or face_roi.shape[1] < self.min_size:
                continue

            is_alive, liveness_reason = self.detector.is_alive(frame, (x, y, x1, y1), face_id=face_id, frame_idx=self.frame_count)
            data["liveness_ok"] = is_alive
            data["liveness_reason"] = liveness_reason

            if self.consent_enabled and is_alive:
                if (self.frame_count - data["last_recognized_frame"] >= self.recognition_interval) and not data["recognition_inflight"]:
                    data["recognition_inflight"] = True
                    threading.Thread(target=recognition_worker, args=(self.recognizer, face_roi, data["recognition_queue"]), daemon=True).start()
                    data["last_recognized_frame"] = self.frame_count

                if self.frame_count % self.emotion_update_interval == 0 and not data["emotion_inflight"]:
                    data["emotion_inflight"] = True
                    threading.Thread(target=emotion_worker, args=(self.emotion_analyzer, face_roi, data["emotion_queue"]), daemon=True).start()

            try:
                status, new_user_id = data["recognition_queue"].get_nowait()
                data["recognition_inflight"] = False
                if self.consent_enabled and status == "ok" and new_user_id != "Unknown":
                    data["user_id"] = new_user_id
            except queue.Empty:
                pass

            try:
                status, emotion = data["emotion_queue"].get_nowait()
                data["emotion_inflight"] = False
                if self.consent_enabled and status == "ok":
                    if isinstance(emotion, dict):
                        label = emotion.get("label", "Unknown")
                        conf = float(emotion.get("confidence", 0.0))
                    else:
                        label = emotion
                        conf = 0.0

                    self.emotion_window.append((label, conf))
                    if len(self.emotion_window) > self.emotion_window_size:
                        self.emotion_window = self.emotion_window[-self.emotion_window_size:]

                    # Rolling majority vote
                    labels = [l for l, c in self.emotion_window if l != "Unknown"]
                    if labels:
                        label = max(set(labels), key=labels.count)
                        conf = max(c for l, c in self.emotion_window if l == label) if self.emotion_window else conf
                    data["emotion"] = label
                    self.emotion_confidence = conf
            except queue.Empty:
                pass

            if (not self.consent_enabled) or (self.blur_unknown and data["user_id"] == "Unknown"):
                blur_region(frame, (x, y, x1, y1))

            if self.consent_enabled and data["user_id"] != data["last_logged_user_id"]:
                log_event({
                    "event": "recognized",
                    "user_id": data["user_id"],
                    "face_id": face_id,
                    "emotion": data["emotion"],
                    "liveness": data["liveness_reason"]
                })
                with self.metrics_lock:
                    self.recognition_events.append(time.time())
                data["last_logged_user_id"] = data["user_id"]

            status_faces.append({
                "id": face_id,
                "user_id": data["user_id"],
                "emotion": data["emotion"],
                "liveness": data["liveness_reason"]
            })

            if data["user_id"] == "Unknown":
                if face_id not in self.unknown_labels:
                    self.unknown_labels[face_id] = f"Unknown {self.unknown_counter}"
                    self.unknown_counter += 1
                if face_id not in self.unknown_first_seen:
                    self.unknown_first_seen[face_id] = time.time()
            else:
                self.unknown_first_seen.pop(face_id, None)

        self.last_fps = 1.0 / max(0.001, (time.time() - start_time))
        self.frame_count += 1
        brightness = float(np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))) if frame.size else 0.0
        face_ratio = (largest_face_area / frame_area) if frame_area else 0.0
        if brightness < 60:
            lighting_hint = "Low light"
        elif brightness > 190:
            lighting_hint = "Too bright"
        else:
            lighting_hint = "Good"
        if face_ratio < 0.08:
            distance_hint = "Move closer"
        elif face_ratio > 0.35:
            distance_hint = "Move back"
        else:
            distance_hint = "Good"

        # Draw dynamic alignment guide for the largest detected face
        if largest_face_area > 0:
            largest = None
            for data in self.face_data.values():
                x, y, x1, y1 = data["box"]
                area = max(0, (x1 - x) * (y1 - y))
                if area == largest_face_area:
                    largest = (x, y, x1, y1)
                    break
            if largest is not None:
                x, y, x1, y1 = largest
                cx = (x + x1) // 2
                cy = (y + y1) // 2
                radius = int(max(30, min(x1 - x, y1 - y) * 0.55))
                cv2.circle(frame, (cx, cy), radius, (63, 182, 255), 2)
                cv2.line(frame, (cx - radius, cy), (cx + radius, cy), (63, 182, 255), 1)
                cv2.line(frame, (cx, cy - radius), (cx, cy + radius), (63, 182, 255), 1)
                cv2.putText(frame, "Align in circle", (cx - radius, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (63, 182, 255), 1)

        self.last_status = {
            "consent": self.consent_enabled,
            "blur_unknown": self.blur_unknown,
            "fps": round(self.last_fps, 1),
            "faces": status_faces,
            "calibration": {
                "brightness": round(brightness, 1),
                "lighting": lighting_hint,
                "face_ratio": round(face_ratio, 3),
                "distance": distance_hint
            },
            "emotion_confidence": round(self.emotion_confidence, 2),
            "unknowns": []
        }
        now = time.time()
        for face_id, t0 in list(self.unknown_first_seen.items()):
            if now - t0 >= 5.0:
                label = self.unknown_labels.get(face_id, "Unknown")
                self.last_status["unknowns"].append({"id": face_id, "label": label})
        # cleanup missing faces
        active_ids = {f["id"] for f in status_faces}
        for face_id in list(self.unknown_first_seen.keys()):
            if face_id not in active_ids:
                self.unknown_first_seen.pop(face_id, None)
        for face_id in list(self.unknown_labels.keys()):
            if face_id not in active_ids:
                self.unknown_labels.pop(face_id, None)
        self.history.append((time.time(), len(status_faces)))
        if len(self.history) > 300:
            self.history = self.history[-300:]
        return frame

    def maybe_process(self, frame, min_interval=0.06):
        now = time.time()
        if now - self.last_process_time < min_interval and self.last_annotated is not None:
            return self.last_annotated
        if not self.process_lock.acquire(blocking=False):
            return self.last_annotated if self.last_annotated is not None else frame
        try:
            annotated = self.process_frame(frame)
            self.last_annotated = annotated
            self.last_process_time = now
            return annotated
        finally:
            self.process_lock.release()

    def get_metrics(self):
        now = time.time()
        window = 120
        with self.metrics_lock:
            self.recognition_events = [t for t in self.recognition_events if now - t <= window]
            buckets = []
            for i in range(12):
                start = now - (11 - i) * 10
                end = start + 10
                count = sum(1 for t in self.recognition_events if start <= t < end)
                buckets.append(count)
        return {"window_sec": window, "bucket_sec": 10, "counts": buckets}

    def get_timeline(self, window_sec=60, bucket_sec=5):
        now = time.time()
        buckets = []
        bucket_count = int(window_sec / bucket_sec)
        for i in range(bucket_count):
            start = now - (bucket_count - 1 - i) * bucket_sec
            end = start + bucket_sec
            count = 0
            for t, face_count in self.history:
                if start <= t < end:
                    count = max(count, face_count)
            buckets.append(count)
        return {"window_sec": window_sec, "bucket_sec": bucket_sec, "counts": buckets}

def recognition_worker(recognizer, face_roi, result_queue):
    try:
        user_id = recognizer.recognize(face_roi)
        result_queue.put(("ok", user_id))
    except Exception as e:
        print(f"Recognition worker error: {e}")
        result_queue.put(("error", "Unknown"))

def emotion_worker(emotion_analyzer, face_roi, result_queue):
    try:
        label, confidence = emotion_analyzer.detect_emotion(face_roi, return_confidence=True)
        result_queue.put(("ok", {"label": label, "confidence": confidence}))
    except Exception as e:
        print(f"Emotion worker error: {e}")
        result_queue.put(("error", {"label": "Unknown", "confidence": 0.0}))

def get_camera_index():
    try:
        return int(os.environ.get("CAM_INDEX", "0"))
    except Exception:
        return 0

camera = CameraStream(get_camera_index())
pipeline = FacePipeline()

@app.route("/")
def index():
    return render_template("index.html", title=WINDOW_TITLE)

@app.route("/video")
def video_feed():
    def gen():
        while True:
            try:
                frame, _ = camera.get_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue
                annotated = pipeline.maybe_process(frame)
                ret, jpeg = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
                if not ret:
                    continue
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n")
            except Exception as e:
                log_line(f"Video generator error: {e}")
                log_line(traceback.format_exc())
                time.sleep(0.1)
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/status")
def api_status():
    payload = dict(pipeline.last_status)
    payload["enroll"] = pipeline.enroll_state
    return jsonify(payload)

@app.route("/api/quote")
def api_quote():
    emotion = request.args.get("emotion", "neutral").strip().lower()
    if emotion in ("unknown", ""):
        emotion = "neutral"
    quote = generate_quote(emotion)
    return jsonify({"emotion": emotion, "quote": quote})

@app.route("/api/enrollments")
def api_enrollments():
    return jsonify({"users": pipeline.recognizer.list_users()})

@app.route("/api/metrics")
def api_metrics():
    return jsonify(pipeline.get_metrics())

@app.route("/api/timeline")
def api_timeline():
    return jsonify(pipeline.get_timeline())

@app.route("/api/model_status")
def api_model_status():
    return jsonify({
        "dlib_shape_predictor": pipeline.recognizer.shape_predictor is not None,
        "dlib_recognition_model": pipeline.recognizer.rec_model is not None,
        "emotion_model": getattr(pipeline.emotion_analyzer, "model", None) is not None
    })

@app.route("/api/audit")
def api_audit():
    if not os.path.exists(EVENT_PATH):
        return Response("", mimetype="text/plain")
    with open(EVENT_PATH, "r", encoding="utf-8") as f:
        data = f.read()
    return Response(data, mimetype="text/plain")

@app.route("/api/consent", methods=["POST"])
def api_consent():
    payload = request.get_json(silent=True) or {}
    value = payload.get("value")
    if value is None:
        pipeline.set_consent(not pipeline.consent_enabled)
    else:
        pipeline.set_consent(bool(value))
    return jsonify({"consent": pipeline.consent_enabled})

@app.route("/api/blur", methods=["POST"])
def api_blur():
    payload = request.get_json(silent=True) or {}
    value = payload.get("value")
    if value is None:
        pipeline.set_blur(not pipeline.blur_unknown)
    else:
        pipeline.set_blur(bool(value))
    return jsonify({"blur_unknown": pipeline.blur_unknown})

@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name required"}), 400

    # Guard for missing dlib model files
    if getattr(pipeline.recognizer, "shape_predictor", None) is None or getattr(pipeline.recognizer, "rec_model", None) is None:
        return jsonify({
            "ok": False,
            "error": "Required dlib models missing. Add shape_predictor_68_face_landmarks.dat and dlib_face_recognition_resnet_model_v1.dat to the project folder."
        }), 400

    def get_next_frame():
        for _ in range(60):
            frame, _ = camera.get_frame()
            if frame is not None:
                return frame
            time.sleep(0.05)
        return None

    if not pipeline.enroll_lock.acquire(blocking=False):
        return jsonify({"ok": False, "error": "Enrollment already in progress"}), 409
    def progress_cb(status, message):
        pipeline.enroll_state = {"status": status, "message": message}

    def run_enroll():
        try:
            pipeline.enroll_state = {"status": "running", "message": "Starting enrollment..."}
            ok = pipeline.recognizer.enroll(get_next_frame, name, progress_cb=progress_cb)
            if ok:
                pipeline.enroll_state = {"status": "done", "message": f"Enrollment complete for {name}."}
            else:
                if pipeline.enroll_state.get("status") != "error":
                    pipeline.enroll_state = {"status": "error", "message": "Enrollment failed."}
        except Exception as e:
            log_line(f"Enroll error: {e}")
            log_line(traceback.format_exc())
            pipeline.enroll_state = {"status": "error", "message": "Enrollment failed (see server.log)."}
        finally:
            pipeline.enroll_lock.release()

    threading.Thread(target=run_enroll, daemon=True).start()
    return jsonify({"ok": True, "status": "started"})

@app.route("/api/enrollments/delete", methods=["POST"])
def api_delete_enrollment():
    payload = request.get_json(silent=True) or {}
    name = (payload.get("name") or "").strip()
    if not name:
        return jsonify({"ok": False, "error": "Name required"}), 400
    ok = pipeline.recognizer.delete_user(name)
    return jsonify({"ok": bool(ok)})

@app.route("/api/camera", methods=["POST"])
def api_camera():
    payload = request.get_json(silent=True) or {}
    cam_index = payload.get("index")
    if cam_index is None:
        return jsonify({"ok": False, "error": "index required"}), 400
    try:
        cam_index = int(cam_index)
    except Exception:
        return jsonify({"ok": False, "error": "index must be int"}), 400
    camera.switch(cam_index)
    return jsonify({"ok": True, "index": cam_index})

def main():
    camera.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)

if __name__ == "__main__":
    main()
