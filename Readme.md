# Advance Face Biometric System

Professional, real‑time biometric platform with face recognition, emotion analysis, liveness checks, audit logging, and a modern web dashboard.

---

## Highlights
- Real‑time **face detection** and **recognition**
- **Emotion detection** using a pretrained model (`emotion_model.h5`)
- **Liveness** checks to reduce spoofing
- **Consent‑first** UI and privacy controls
- Enrollment management (list, search, delete)
- Audit log export and operational metrics
- Gemini‑powered **Good Vibe Quotes** based on emotion
- USB camera support via UI selector

---

## Screenshots
Replace the image paths with your two screenshots (already in the project).

![Dashboard](Screenshot 2026-02-06 002357.png)
![Enrollment](screenshots/ui-2.png)

---

## Tech Stack
- Python 3.10+
- OpenCV (video + detection)
- dlib (face embeddings)
- TensorFlow/Keras (emotion model)
- Flask (web dashboard)
- Gemini API (emotion‑aware quotes)

---

## Project Structure
```
AdvanceFaceBiometricSystem/
├── web_app.py                  # Web dashboard server
├── main.py                     # Optional CLI app
├── face_detector.py
├── face_recognizer.py
├── emotion_analyzer.py
├── utils.py
├── templates/
│   └── index.html
├── static/
│   ├── app.css
│   └── app.js
├── emotion_model.h5            # Pretrained emotion model
├── haarcascade_frontalface_default.xml
├── shape_predictor_68_face_landmarks.dat
├── dlib_face_recognition_resnet_model_v1.dat
├── events.jsonl                # Audit events
└── requirements.txt
```

---

## Requirements
### Hardware
- Webcam or USB camera
- 8GB+ RAM recommended

### Software
- Python 3.10 or newer
- Windows 10/11, Linux, or macOS
- **CMake + Visual Studio Build Tools** (for dlib on Windows)

---

## Installation
### 1. Create and activate venv
```bash
python -m venv venv
venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> On Windows, `dlib` requires CMake and Visual C++ Build Tools.

---

## Models (Required Files)
Make sure these files exist in the **root directory**:
- `emotion_model.h5`
- `haarcascade_frontalface_default.xml`
- `shape_predictor_68_face_landmarks.dat`
- `dlib_face_recognition_resnet_model_v1.dat`

If any are missing, recognition or emotion analysis will be disabled.

---

## Gemini Quotes Setup
Set your API key in the same terminal session before running:

**PowerShell**
```bash
$env:GEMINI_API_KEY="YOUR_KEY"
```

**CMD**
```bash
set GEMINI_API_KEY=YOUR_KEY
```

---

## Run (Web Dashboard)
```bash
python web_app.py
```
Open:
```
http://127.0.0.1:5000
```

---

## Enrollment Workflow
1. Turn **Consent ON**
2. Unknown faces appear in the **Unknown Queue** after 5 seconds
3. Click **Use** to prefill the name
4. Click **Enroll**

---

## USB Camera
Use the **Camera selector** in the UI.  
Try **Camera 1** for most USB webcams.

---

## Audit Log
All recognition events are logged in:
```
events.jsonl
```
Use the **Export Audit Log** button in the UI to download the file.

---

## Troubleshooting
**Emotion always Unknown**
- Ensure `emotion_model.h5` is present
- Turn **Consent ON**
- Check **Model Status** panel

**Dlib errors**
- Confirm CMake is in PATH
- Install Visual C++ Build Tools

**Camera fails or freezes**
- Switch camera index in the UI
- Replug the USB camera

---

## Dataset (FER2013)
Use the FER2013 dataset to retrain the emotion model:
```
https://www.kaggle.com/datasets/msambare/fer2013
```

---

## License
MIT License
