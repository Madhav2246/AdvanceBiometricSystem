# Advance Face Biometric System

<p align="center">
  <img src="https://img.shields.io/badge/Biometric-Platform-00c2ff?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Face-Recognition-6c5ce7?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Emotion-AI-00d084?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Consent-First-ff6b6b?style=for-the-badge" />
</p>

**Professional real‑time biometric system** with face recognition, emotion analysis, liveness checks, audit logging, and a modern web dashboard.

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

![Dashboard](screenshots/ui-1.png)
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

## ⚙️ Setup
### 1) Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> On Windows, `dlib` requires **CMake** and **Visual C++ Build Tools**.

---

## 🔑 Gemini Quotes (Optional)
Set API key in the same terminal before running:

**PowerShell**
```bash
$env:GEMINI_API_KEY="YOUR_KEY"
```

**CMD**
```bash
set GEMINI_API_KEY=YOUR_KEY
```

---

## ▶️ Run the Web Dashboard
```bash
python web_app.py
```
Open:
```
http://127.0.0.1:5000
```

---

## ✅ Enrollment Workflow
1. Turn **Consent ON**
2. Unknown faces appear in the **Unknown Queue** after 5 seconds
3. Click **Use** to prefill the name
4. Click **Enroll**

---

## 📷 USB Camera
Use the **Camera selector** in the UI.  
Try **Camera 1** for most USB webcams.

---

## 🗂️ Audit Logs
All recognition events are written to:
```
events.jsonl
```
Use the **Export Audit Log** button to download the file.

---

## 🧪 Troubleshooting
**Emotion always `Unknown`**
- Ensure `emotion_model.h5` is present
- Consent must be **ON**
- Check the **Model Status** panel

**Dlib install errors**
- Confirm CMake is in PATH
- Install Visual C++ Build Tools

**Camera fails**
- Switch camera index in the UI
- Replug USB camera

---

## 📁 Project Structure
```
AdvanceFaceBiometricSystem/
├── web_app.py
├── main.py
├── face_detector.py
├── face_recognizer.py
├── emotion_analyzer.py
├── utils.py
├── templates/
│   └── index.html
├── static/
│   ├── app.css
│   └── app.js
├── requirements.txt
└── Readme.md
```

---

## 📌 Dataset
FER2013 (for training `emotion_model.h5`):  
`https://www.kaggle.com/datasets/msambare/fer2013`

---

## 📜 License
MIT
