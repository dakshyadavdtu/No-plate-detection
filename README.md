# 🚗 License Plate Detection App using YOLOv8 + EasyOCR + Streamlit

This is a complete web app built with **Streamlit**, using **YOLOv8** for vehicle and license plate detection, and **EasyOCR** for recognizing the license text.

---

## 📦 Features

- Detects cars/bikes in uploaded images or videos
- Detects and crops license plates using a fine-tuned YOLOv8 model
- Reads license numbers using EasyOCR
- Saves results (plate crops + license text + metadata) into a downloadable CSV
- Supports:
  - 📸 Image Upload
  - 🎥 Video Upload
  - 📷 Live Camera Detection
  - 📡 RTSP Stream Detection
- Built with modular design for real-time and offline inference

---

## 🧠 Models Used

- `yolov8n.pt`: COCO model for detecting vehicles
- `license_plate_detector.pt`: Fine-tuned YOLOv8 model for license plate detection

Place both models inside the `uploads/` folder.

---

## 📁 Project Structure

