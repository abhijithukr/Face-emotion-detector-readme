# 😶‍🌫️ Face Emotion Detector

> Real-time facial emotion recognition using OpenCV and DeepFace — detects 7 emotions from webcam or image input.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

---

## 🎯 What It Does

Detects and classifies human facial emotions in real-time using your webcam or any input image. Recognizes 7 emotions:

`Happy` · `Sad` · `Angry` · `Surprised` · `Fearful` · `Disgusted` · `Neutral`

## 🖥️ Demo

![Demo GIF](assets/demo.gif)

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **OpenCV** — face detection & webcam capture
- **DeepFace** — pre-trained emotion classification model
- **NumPy** — array processing

---

## 🚀 Getting Started

```bash
# Clone the repo
git clone https://github.com/abhijithukr/face-emotion-detector
cd face-emotion-detector

# Install dependencies
pip install -r requirements.txt

# Run on webcam
python detect.py --source webcam

# Run on an image
python detect.py --source image --path ./samples/test.jpg
```

---

## 📁 Project Structure

```
face-emotion-detector/
├── detect.py          # Main detection script
├── model/             # Pre-trained model weights
├── samples/           # Sample test images
├── requirements.txt
└── README.md
```

---

## 📦 Requirements

```
opencv-python
deepface
numpy
tf-keras
```

---

## 🔮 Future Improvements

- [ ] Add emotion history graph over time
- [ ] Multi-face detection support
- [ ] Web interface using Flask

---

## 📄 License

MIT License © 2024 [abhijithukr](https://github.com/abhijithukr)
