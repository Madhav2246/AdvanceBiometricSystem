# Advanced Face Biometric System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python Version" />
  <img src="https://img.shields.io/badge/OpenCV-4.x-green.svg" alt="OpenCV" />
  <img src="https://img.shields.io/badge/TensorFlow-2.x-orange.svg" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Biometric-Platform-00c2ff?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Face-Recognition-6c5ce7?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Emotion-AI-00d084?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Consent-First-ff6b6b?style=for-the-badge" />
</p>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Screenshots](#-screenshots)
- [Technology Stack](#-technology-stack)
- [System Requirements](#-system-requirements)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [API Documentation](#-api-documentation)
- [Troubleshooting](#-troubleshooting)
- [Security & Privacy](#-security--privacy)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 Overview

**Advanced Face Biometric System** is a professional-grade, real-time biometric platform designed with privacy and user consent at its core. The system combines cutting-edge computer vision, deep learning, and modern web technologies to deliver accurate face recognition, emotion analysis, and liveness detection capabilities through an intuitive web interface.

Built for security-conscious organizations and developers, this system provides enterprise-level features including comprehensive audit logging, enrollment management, and operational metrics—all while maintaining strict adherence to privacy-first principles.

### Use Cases

- **Access Control**: Secure building entry and restricted area access
- **Attendance Systems**: Automated employee or student attendance tracking
- **Customer Analytics**: Emotion-based customer experience insights (with explicit consent)
- **Security Monitoring**: Real-time threat detection and visitor management
- **Healthcare**: Patient identification and emotional state monitoring

---

## ✨ Key Features

### Core Capabilities

- **🎭 Real-Time Face Detection**: High-performance face detection using Haar Cascade classifiers
- **👤 Face Recognition**: Robust face identification using dlib's ResNet-based embeddings
- **😊 Emotion Analysis**: 7-class emotion detection (Happy, Sad, Angry, Surprised, Neutral, Fear, Disgust)
- **🔒 Liveness Detection**: Anti-spoofing checks to prevent photo/video attacks
- **✅ Consent-First Architecture**: All biometric processing requires explicit user consent

### Management & Monitoring

- **📊 Web Dashboard**: Modern, responsive interface for system management
- **👥 Enrollment Management**: Easy onboarding with search, filter, and bulk operations
- **📝 Comprehensive Audit Logging**: JSONL-format event logging for compliance
- **📈 Operational Metrics**: Real-time system performance and usage statistics
- **🎥 Multi-Camera Support**: USB camera selection and hot-swapping

### Advanced Features

- **💬 AI-Powered Quotes**: Gemini-generated motivational quotes based on detected emotions
- **🔍 Unknown Face Queue**: Automatic detection and enrollment workflow for new faces
- **📤 Data Export**: One-click audit log and metrics export
- **⚡ Real-Time Processing**: Sub-100ms inference time on modern hardware

---

## 📸 Screenshots

<div align="center">

### Main Dashboard
![Dashboard Overview](https://raw.githubusercontent.com/Madhav2246/AdvanceBiometricSystem/main/Screenshot%202026-02-06%20002357.png)
*Real-time recognition with emotion analysis and consent controls*

### Enrollment Interface
![Enrollment Management](https://raw.githubusercontent.com/Madhav2246/AdvanceBiometricSystem/main/Screenshot%202026-02-06%20002640.png)
*Streamlined enrollment workflow with unknown face detection*

</div>
---

## 🛠️ Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Backend** | Python 3.10+ | Core processing engine |
| **Web Framework** | Flask 2.x | RESTful API and web server |
| **Computer Vision** | OpenCV 4.x | Video capture and image processing |
| **Face Detection** | Haar Cascades | Real-time face detection |
| **Face Recognition** | dlib | Face embedding and recognition |
| **Emotion AI** | TensorFlow/Keras | Deep learning emotion classification |
| **AI Integration** | Google Gemini API | Context-aware quote generation |
| **Frontend** | HTML5/CSS3/JavaScript | Responsive web interface |

### Key Libraries

```
opencv-python>=4.8.0
dlib>=19.24.0
tensorflow>=2.13.0
keras>=2.13.0
Flask>=2.3.0
numpy>=1.24.0
Pillow>=10.0.0
google-generativeai>=0.3.0
```

---

## 💻 System Requirements

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **Processor** | Intel Core i5 / AMD Ryzen 5 | Intel Core i7 / AMD Ryzen 7 |
| **RAM** | 8 GB | 16 GB |
| **Storage** | 2 GB free space | 5 GB free space |
| **Camera** | 720p webcam | 1080p webcam |
| **GPU** | Not required | NVIDIA GPU (CUDA support) |

### Software Requirements

- **Operating System**: Windows 10/11, Ubuntu 20.04+, macOS 11+
- **Python**: 3.10 or newer
- **Build Tools** (Windows): 
  - CMake 3.15+
  - Visual Studio Build Tools 2019/2022
  - C++ compiler

### Model Files Required

Ensure these files are present in the project root:

```
emotion_model.h5                              # Emotion classification model
haarcascade_frontalface_default.xml           # Face detection cascade
shape_predictor_68_face_landmarks.dat         # Facial landmarks predictor
dlib_face_recognition_resnet_model_v1.dat     # Face recognition model
```

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/AdvanceFaceBiometricSystem.git
cd AdvanceFaceBiometricSystem
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note for Windows Users:**

If `dlib` installation fails, install build tools first:

1. Install [CMake](https://cmake.org/download/)
2. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
3. Select "Desktop development with C++" workload
4. Retry: `pip install dlib`

### 4. Download Model Files

Download required model files:

```bash
# Emotion model (if not included)
# Download from your trained model repository

# dlib models
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2

wget http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2
bunzip2 dlib_face_recognition_resnet_model_v1.dat.bz2
```

### 5. Verify Installation

```bash
python -c "import cv2, dlib, tensorflow; print('All dependencies installed successfully!')"
```

---

## ⚙️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# Gemini API Configuration (Optional)
GEMINI_API_KEY=your_gemini_api_key_here

# Flask Configuration
FLASK_ENV=production
FLASK_DEBUG=False
SECRET_KEY=your_secret_key_here

# Camera Configuration
DEFAULT_CAMERA_INDEX=0

# System Configuration
MAX_ENROLLMENT_SIZE=1000
RECOGNITION_THRESHOLD=0.6
LIVENESS_CHECK_ENABLED=True
```

### Setting API Keys

**PowerShell (Windows):**
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

**CMD (Windows):**
```cmd
set GEMINI_API_KEY=YOUR_API_KEY
```

**Bash (Linux/macOS):**
```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

To make it permanent, add to `.bashrc` or `.zshrc`:
```bash
echo 'export GEMINI_API_KEY="YOUR_API_KEY"' >> ~/.bashrc
source ~/.bashrc
```

### Camera Configuration

Edit camera settings in `web_app.py` if needed:

```python
# Default camera index (0 = built-in, 1 = USB)
CAMERA_INDEX = 0

# Camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Frame rate
CAMERA_FPS = 30
```

---

## 🚀 Usage

### Starting the Web Dashboard

```bash
python web_app.py
```

The dashboard will be accessible at:
```
http://127.0.0.1:5000
```

### Enrollment Workflow

1. **Enable Consent**: Toggle the consent switch to "ON"
2. **Position Face**: Ensure face is clearly visible in the camera feed
3. **Auto-Detection**: Unknown faces appear in the "Unknown Queue" after 5 seconds
4. **Enroll User**: 
   - Click "Use" to prefill the name field
   - Modify name if needed
   - Click "Enroll" to register the face

### Recognition Process

1. **Consent Active**: Ensure consent toggle is "ON"
2. **Face Detection**: System automatically detects faces in frame
3. **Recognition**: Matches against enrolled faces
4. **Emotion Analysis**: Displays current emotional state
5. **Liveness Check**: Validates face is live (not photo/video)

### Camera Selection

Use the **Camera Selector** dropdown in the UI:
- **Camera 0**: Built-in webcam
- **Camera 1**: First USB camera
- **Camera 2**: Second USB camera (if available)

### Exporting Data

**Audit Logs:**
1. Click "Export Audit Log" button
2. Download `events.jsonl` file

**Metrics:**
1. View real-time metrics in the dashboard
2. Export via browser console if needed

---

## 📁 Project Structure

```
AdvanceFaceBiometricSystem/
│
├── 📄 main.py                              # CLI application (optional)
├── 🌐 web_app.py                           # Flask web server & API
├── 🎭 face_detector.py                     # Face detection module
├── 👤 face_recognizer.py                   # Face recognition engine
├── 😊 emotion_analyzer.py                  # Emotion classification
├── 🛠️ utils.py                             # Utility functions
│
├── 📁 templates/
│   └── index.html                          # Main dashboard template
│
├── 📁 static/
│   ├── app.css                             # Custom styles
│   └── app.js                              # Frontend logic
│
├── 📁 screenshots/
│   ├── ui-1.png                            # Dashboard screenshot
│   └── ui-2.png                            # Enrollment screenshot
│
├── 🤖 emotion_model.h5                     # Emotion classification model
├── 📊 haarcascade_frontalface_default.xml  # Face detection cascade
├── 🎯 shape_predictor_68_face_landmarks.dat # Facial landmarks
├── 🧠 dlib_face_recognition_resnet_model_v1.dat # Face embeddings
│
├── 📝 events.jsonl                         # Audit log (generated)
├── 📋 requirements.txt                     # Python dependencies
├── 📖 README.md                            # This file
└── 📜 LICENSE                              # MIT License

```

---

## 🔌 API Documentation

### REST Endpoints

#### Video Feed
```http
GET /video_feed
```
Returns MJPEG stream for live video display.

#### Recognition Status
```http
GET /recognition_status
```
Returns current recognition state and detected faces.

**Response:**
```json
{
  "recognized": true,
  "name": "John Doe",
  "emotion": "Happy",
  "confidence": 0.95,
  "timestamp": "2024-02-06T10:30:00Z"
}
```

#### Enrollment
```http
POST /enroll
Content-Type: application/json

{
  "name": "Jane Smith"
}
```

#### Get Enrollments
```http
GET /enrollments
```

#### Delete Enrollment
```http
DELETE /enrollments/<name>
```

---

## 🐛 Troubleshooting

### Common Issues

#### Emotion Always Shows "Unknown"

**Symptoms:** Emotion field remains "Unknown" despite face detection

**Solutions:**
1. Verify `emotion_model.h5` exists in project root
2. Ensure consent is enabled (toggle ON)
3. Check Model Status panel for errors
4. Verify TensorFlow installation: `python -c "import tensorflow; print(tensorflow.__version__)"`

#### Dlib Installation Errors (Windows)

**Symptoms:** `pip install dlib` fails with compilation errors

**Solutions:**
1. Install CMake: `choco install cmake` or download from cmake.org
2. Install Visual Studio Build Tools with C++ workload
3. Ensure CMake is in system PATH
4. Try: `pip install dlib-binary` (precompiled wheel)
5. Alternative: `pip install cmake` then `pip install dlib`

#### Camera Not Detected

**Symptoms:** Black screen or "Camera failed to initialize"

**Solutions:**
1. Try different camera indices (0, 1, 2) in UI selector
2. Replug USB camera
3. Check camera permissions in OS settings
4. Test camera: `python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"`
5. Close other applications using the camera (Zoom, Skype, etc.)

#### High CPU Usage

**Symptoms:** System lag, high processor utilization

**Solutions:**
1. Reduce camera resolution in `web_app.py`
2. Lower frame rate (default 30fps → 15fps)
3. Enable GPU acceleration (if NVIDIA GPU available)
4. Limit concurrent recognition requests

#### Face Not Recognized

**Symptoms:** Enrolled face shows as "Unknown"

**Solutions:**
1. Re-enroll with better lighting conditions
2. Ensure face is directly facing camera
3. Adjust recognition threshold in configuration
4. Verify enrollment was successful (check enrollments list)

---

## 🔐 Security & Privacy

### Privacy Principles

1. **Consent-First**: All biometric processing requires explicit user consent
2. **Data Minimization**: Only essential biometric data is stored
3. **Transparent Logging**: All operations are logged for audit trails
4. **User Control**: Users can delete their biometric data anytime

### Security Best Practices

- **Production Deployment**: Never expose Flask dev server to public internet
- **Use HTTPS**: Deploy behind reverse proxy (nginx/Apache) with SSL
- **Authentication**: Implement user authentication for dashboard access
- **Rate Limiting**: Add API rate limiting to prevent abuse
- **Data Encryption**: Encrypt `events.jsonl` and biometric data at rest
- **Regular Updates**: Keep dependencies updated for security patches

### GDPR Compliance Considerations

- Implement data retention policies
- Provide data export functionality
- Enable right-to-deletion (already implemented)
- Document data processing activities
- Obtain explicit consent before enrollment

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

### How to Contribute

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/AmazingFeature`
3. Commit changes: `git commit -m 'Add AmazingFeature'`
4. Push to branch: `git push origin feature/AmazingFeature`
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guide for Python code
- Add unit tests for new features
- Update documentation for API changes
- Ensure all tests pass before submitting PR

### Code of Conduct

Please note we have a code of conduct. Follow it in all interactions with the project.

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Advanced Face Biometric System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

### Datasets & Models

- **FER-2013 Dataset**: Emotion recognition training data  
  [Kaggle FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
  
- **dlib Models**: Pre-trained face recognition models  
  Davis King - [dlib.net](http://dlib.net/)

### Technologies & Libraries

- OpenCV team for computer vision tools
- TensorFlow team for deep learning framework
- Flask team for web framework
- Google for Gemini API

### Inspiration

This project was inspired by the need for privacy-respecting, open-source biometric solutions that prioritize user consent and transparency.

---

## 📞 Support & Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/AdvanceFaceBiometricSystem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/AdvanceFaceBiometricSystem/discussions)
- **Email**: your.email@example.com

---

## 🗺️ Roadmap

### Upcoming Features

- [ ] Multi-face recognition in single frame
- [ ] Face mask detection
- [ ] Age and gender estimation
- [ ] Cloud storage integration
- [ ] Mobile app (iOS/Android)
- [ ] Docker containerization
- [ ] Kubernetes deployment support
- [ ] Advanced analytics dashboard
- [ ] REST API authentication
- [ ] Webhook notifications

---

<div align="center">

### ⭐ Star this repository if you find it useful!

Made with ❤️ by developers who care about privacy

[Report Bug](https://github.com/yourusername/AdvanceFaceBiometricSystem/issues) · [Request Feature](https://github.com/yourusername/AdvanceFaceBiometricSystem/issues) · [Documentation](https://github.com/yourusername/AdvanceFaceBiometricSystem/wiki)

</div>
