# 🤟 ASL Recognition System
### Real-Time American Sign Language Letters and Numbers Recognition Using Image Processing

A real-time system that recognizes American Sign Language (ASL) letters and numbers using **image processing** and **deep learning**.  
The project aims to make communication easier between hearing and hearing-impaired individuals through AI-powered recognition.

---

## 🧠 Abstract
This project presents a **real-time American Sign Language (ASL) recognition system** designed to help people who do not know sign language communicate easily with hearing-impaired individuals.

A **CNN-based model** was trained using a **custom dataset** consisting of approximately **350–400 images** for each letter (A–Z) and number (0–9).  
The dataset includes a wide variety of **lighting conditions, skin tones, and backgrounds** to ensure model diversity and accuracy.  
After training, the model was optimized using **TensorFlow Lite** for faster and more efficient real-time performance.

---

## 🗂️ Dataset
A custom dataset was created by capturing hand gesture images using a webcam.  
Each letter and number has its own folder containing 350–400 labeled images.
![Example Output](https://github.com/nazozer/ASL-Recognition/blob/main/B_letter.png)
!(https://github.com/nazozer/ASL-Recognition/blob/main/Number_6.png)

---

## 🧩 Technologies Used

| Tool / Library | Purpose |
| --- | --- |
| **Python** | Main programming language |
| **TensorFlow / Keras** | Model training and evaluation |
| **TensorFlow Lite** | Model optimization for real-time performance |
| **OpenCV (cv2)** | Image processing and video capture |
| **CVZone** | Hand tracking and gesture detection |
| **NumPy** | Numerical and matrix operations |

---

## 💻 Features

- Real-time webcam input  
- Recognition of ASL **letters (A–Z)** and **numbers (0–9)**  
- **Interactive typing functions:**  
  - **Enter** → Add letter  
  - **Space** → Add space  
  - **Backspace** → Delete last character  

---

## 🧠 Results

The system successfully detects and classifies ASL gestures in real time.  
Predictions are displayed on the screen, allowing users to form words and numbers interactively.  

> Example: When you show the hand sign for “A”, the system recognizes it and displays “A” on the screen instantly.

---
