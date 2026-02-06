import cv2
import time
import threading
import queue
import numpy as np
import os
import json
from datetime import datetime
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from emotion_analyzer import EmotionAnalyzer

WINDOW_NAME = "Face Biometric System"

def recognition_worker(recognizer, face_roi, result_queue):
    try:
        user_id = recognizer.recognize(face_roi)
        result_queue.put(("ok", user_id))
    except Exception as e:
        print(f"Recognition worker error: {e}")
        result_queue.put(("error", "Unknown"))

def emotion_worker(emotion_analyzer, face_roi, result_queue):
    try:
        emotion = emotion_analyzer.detect_emotion(face_roi)
        result_queue.put(("ok", emotion))
    except Exception as e:
        print(f"Emotion worker error: {e}")
        result_queue.put(("error", "Unknown"))

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

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

def blur_region(frame, box, ksize=31):
    x, y, x1, y1 = box
    x, y, x1, y1 = max(0, x), max(0, y), min(frame.shape[1], x1), min(frame.shape[0], y1)
    if x1 <= x or y1 <= y:
        return
    roi = frame[y:y1, x:x1]
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    frame[y:y1, x:x1] = cv2.GaussianBlur(roi, (ksize, ksize), 0)

def draw_overlay(frame, consent_enabled, blur_unknown, fps):
    lines = [
        "Face Biometric System | q: quit | c: consent | b: blur | e: enroll",
        f"Consent: {'ON' if consent_enabled else 'OFF'} | Blur unknown: {'ON' if blur_unknown else 'OFF'} | FPS: {fps:.1f}"
    ]
    y = 18
    for line in lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 18

def log_event(event_path, payload):
    try:
        payload = dict(payload)
        payload["timestamp"] = now_iso()
        with open(event_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")
    except Exception as e:
        print(f"Event log error: {e}")

def choose_enrollment_target(face_data, frame_shape):
    if not face_data:
        return None
    h, w = frame_shape[:2]
    center = (w / 2, h / 2)
    best_id = None
    best_dist = float("inf")
    for face_id, data in face_data.items():
        if data["user_id"] != "Unknown":
            continue
        x, y, x1, y1 = data["box"]
        cx = (x + x1) / 2
        cy = (y + y1) / 2
        dist = (cx - center[0]) ** 2 + (cy - center[1]) ** 2
        if dist < best_dist:
            best_dist = dist
            best_id = face_id
    return best_id

def main():
    cap = None
    try:
        print("Preloading models...")
        detector = FaceDetector()
        recognizer = FaceRecognizer()
        emotion_analyzer = EmotionAnalyzer()
        print("Models preloaded.")

        # Prefer DirectShow backend on Windows for better stability
        cam_index = int(os.environ.get("CAM_INDEX", "0"))
        cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print("Error: Could not open webcam.")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"Webcam resolution set to: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")
        print(f"Webcam FPS set to: {cap.get(cv2.CAP_PROP_FPS)}")

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        print("Camera window created.")

        frame_count = 0
        detection_interval = 4
        recognition_interval = 12
        emotion_update_interval = 10
        max_missed_frames = 20
        iou_threshold = 0.4
        padding = 40
        min_size = 60
        last_faces = []

        face_data = {}
        next_face_id = 0

        consent_enabled = False
        blur_unknown = True
        last_enroll_time = 0.0
        enroll_cooldown_sec = 8.0
        event_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "events.jsonl")

        def get_next_frame_for_enrollment(prompt_text):
            while True:
                ret, frame = cap.read()
                if not ret:
                    return None
                cv2.putText(frame, prompt_text, (10, frame.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(1) & 0xFF
                if key != 255:
                    return frame

        while True:
            start_time = time.time()

            ret, frame = cap.read()
            if not ret:
                print("Warning: Could not read frame. Reinitializing camera...")
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
                if not cap.isOpened():
                    print("Error: Could not reopen webcam.")
                    break
                continue

            if frame_count % detection_interval == 0:
                faces = detector.detect_faces(frame)
            else:
                faces = last_faces

            new_face_data = {}
            for face in faces:
                x, y, x1, y1 = face
                best_iou = 0
                best_face_id = None

                for face_id, data in face_data.items():
                    iou = calculate_iou(face, data["box"])
                    if iou > best_iou and iou > iou_threshold:
                        best_iou = iou
                        best_face_id = face_id

                if best_face_id is not None:
                    new_face_data[best_face_id] = face_data[best_face_id]
                    new_face_data[best_face_id]["box"] = (x, y, x1, y1)
                else:
                    new_face_data[next_face_id] = {
                        "box": (x, y, x1, y1),
                        "user_id": "Unknown",
                        "emotion": "Unknown",
                        "recognition_queue": queue.Queue(maxsize=1),
                        "emotion_queue": queue.Queue(maxsize=1),
                        "last_seen": frame_count,
                        "last_recognized_frame": -recognition_interval,
                        "recognition_inflight": False,
                        "emotion_inflight": False,
                        "liveness_ok": False,
                        "liveness_reason": "checking",
                        "last_logged_user_id": None
                    }
                    next_face_id += 1

            face_data = {k: v for k, v in new_face_data.items() if frame_count - v["last_seen"] < max_missed_frames}
            last_faces = [data["box"] for data in face_data.values()]

            for face_id, data in face_data.items():
                x, y, x1, y1 = data["box"]
                data["last_seen"] = frame_count

                if not detector.validate_face(frame, (x, y, x1, y1)):
                    continue

                x_roi = max(0, x - padding)
                y_roi = max(0, y - padding)
                x1_roi = min(frame.shape[1], x1 + padding)
                y1_roi = min(frame.shape[0], y1 + padding)

                face_roi = frame[y_roi:y1_roi, x_roi:x1_roi]
                if face_roi.size == 0 or face_roi.shape[0] < min_size or face_roi.shape[1] < min_size:
                    continue

                is_alive, liveness_reason = detector.is_alive(frame, (x, y, x1, y1), face_id=face_id, frame_idx=frame_count)
                data["liveness_ok"] = is_alive
                data["liveness_reason"] = liveness_reason

                if consent_enabled and is_alive:
                    if (frame_count - data["last_recognized_frame"] >= recognition_interval) and not data["recognition_inflight"]:
                        data["recognition_inflight"] = True
                        threading.Thread(target=recognition_worker, args=(recognizer, face_roi, data["recognition_queue"]), daemon=True).start()
                        data["last_recognized_frame"] = frame_count

                    if frame_count % emotion_update_interval == 0 and not data["emotion_inflight"]:
                        data["emotion_inflight"] = True
                        threading.Thread(target=emotion_worker, args=(emotion_analyzer, face_roi, data["emotion_queue"]), daemon=True).start()

                try:
                    status, new_user_id = data["recognition_queue"].get_nowait()
                    data["recognition_inflight"] = False
                    if consent_enabled and status == "ok" and new_user_id != "Unknown":
                        data["user_id"] = new_user_id
                except queue.Empty:
                    pass

                try:
                    status, emotion = data["emotion_queue"].get_nowait()
                    data["emotion_inflight"] = False
                    if consent_enabled and status == "ok":
                        data["emotion"] = emotion
                except queue.Empty:
                    pass

                label = f"{data['user_id']} - {data['emotion']} | {data['liveness_reason']}"
                if (not consent_enabled) or (blur_unknown and data["user_id"] == "Unknown"):
                    blur_region(frame, (x, y, x1, y1))
                color = (0, 255, 0) if data["user_id"] != "Unknown" else (0, 200, 255)
                cv2.rectangle(frame, (x, y), (x1, y1), color, 1)
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                if consent_enabled and data["user_id"] != data["last_logged_user_id"]:
                    log_event(event_path, {
                        "event": "recognized",
                        "user_id": data["user_id"],
                        "face_id": face_id,
                        "emotion": data["emotion"],
                        "liveness": data["liveness_reason"]
                    })
                    data["last_logged_user_id"] = data["user_id"]

            fps = 1.0 / max(0.001, (time.time() - start_time))
            draw_overlay(frame, consent_enabled, blur_unknown, fps)
            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            if key == ord('c'):
                consent_enabled = not consent_enabled
                log_event(event_path, {"event": "consent_toggled", "consent": consent_enabled})
            if key == ord('b'):
                blur_unknown = not blur_unknown
            if key == ord('e'):
                if not consent_enabled:
                    print("Consent is OFF. Toggle consent with 'c' before enrollment.")
                elif time.time() - last_enroll_time < enroll_cooldown_sec:
                    print("Enrollment is cooling down. Please wait a moment.")
                else:
                    target_id = choose_enrollment_target(face_data, frame.shape)
                    if target_id is None:
                        print("No suitable unknown face found for enrollment.")
                    else:
                        name = input(f"Enter name for Face {target_id}: ").strip()
                        if name:
                            last_enroll_time = time.time()
                            success = recognizer.enroll(lambda: get_next_frame_for_enrollment("Press any key to capture"), name)
                            if success:
                                face_data[target_id]["user_id"] = name.lower()
                                log_event(event_path, {"event": "enrolled", "user_id": name.lower(), "face_id": target_id})
                            else:
                                print(f"Enrollment failed for {name}.")
                        else:
                            print("No name provided. Enrollment canceled.")

            frame_count += 1

    except Exception as e:
        print(f"Main loop error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
