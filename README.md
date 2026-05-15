 # VisionTouch – AI Gesture Control System

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Platform](https://img.shields.io/badge/platform-windows-green.svg)

VisionTouch is an AI-powered gesture control system that allows users to control their computer using real-time hand gestures through a webcam.

The project uses Computer Vision and Machine Learning with MediaPipe and OpenCV to detect hand landmarks and convert them into mouse actions, scrolling, media controls, and system interactions — all without touching the computer.

No external hardware is required.

---

# Features

## Gesture Controls

| Action | Gesture |
|---|---|
| Neutral / Idle | Open Palm 🖐 |
| Move Cursor | Index Finger ☝ |
| Left Click | Thumb + Index Pinch 👌 |
| Right Click | Thumb + Middle Pinch |
| Double Click | Index + Middle Together ✌ |
| Scroll | Two Fingers Up ✌ |
| Drag & Drop | Hold Pinch |
| Multiple Selection | Three Fingers 🖖 |
| Volume Control | Thumb ↔ Index Distance |
| Brightness Control | Thumb ↔ Middle Distance |

---

# Gesture Details

## Neutral Gesture
Open palm gesture used as the idle/safe state.

- Stops accidental actions
- Keeps tracking active
- Improves stability

---

## Move Cursor
Raise only the index finger.

- Cursor follows fingertip movement
- Smoothed movement for better precision
- Real-time tracking

---

## Left Click
Pinch thumb and index finger together.

- Natural interaction
- Quick response
- Reduced accidental clicks

---

## Right Click
Pinch thumb and middle finger together.

- Separate gesture from left click
- Easy landmark detection

---

## Double Click
Bring index and middle fingers together.

- Faster and more reliable than repeated pinching

---

## Scrolling
Raise index and middle fingers.

- Move hand vertically to scroll
- Smooth scrolling experience

---

## Drag and Drop
Hold the pinch gesture.

- Pinch begins drag
- Release ends drag

---

## Multiple Item Selection
Raise three fingers.

- Useful for selecting multiple files/items
- Can simulate CTRL selection behavior

---

## Volume Control
Adjust distance between thumb and index finger.

- Dynamic system volume control
- Visual feedback support

---

## Brightness Control
Adjust distance between thumb and middle finger.

- Dynamic brightness adjustment
- Smooth brightness scaling

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| OpenCV | Webcam capture and image processing |
| MediaPipe | Hand tracking and landmark detection |
| PyAutoGUI | Mouse and keyboard automation |
| NumPy | Mathematical operations |
| Screen Brightness Control | Brightness adjustment |
| Pycaw | Windows audio control |

---

# Requirements

- Python 3.11
- Windows OS
- Webcam

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/VisionTouch.git