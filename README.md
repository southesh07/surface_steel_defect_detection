# surface_steel_defect_detection

An AI-powered web application that automatically detects scratches on metal and steel surfaces using **YOLOv11** object detection, deployed with **Streamlit**.

---

## 📸 Demo

Upload any image of a metal surface and the model will instantly highlight scratches with bounding boxes, confidence scores, and detailed metrics.

---

## ✨ Features

- 📁 **File Upload** — Upload JPG/PNG images directly from your device
- 🌐 **Image URL** — Paste a direct image link to load and analyze it
- 🧠 **CLAHE Enhancement** — Contrast Limited Adaptive Histogram Equalization (OpenCV) to boost faint scratch visibility
- 🎛️ **Adjustable Sensitivity** — Confidence threshold slider (0.05 – 0.10) to tune detection precision
- 📊 **Confidence Metrics** — Displays scratch count, average confidence, and max confidence
- ⚡ **Real-time Results** — Annotated output image with bounding boxes rendered instantly

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| [YOLOv11 (Ultralytics)](https://github.com/ultralytics/ultralytics) | Object detection model |
| [Streamlit](https://streamlit.io/) | Web application framework |
| [OpenCV](https://opencv.org/) | Image preprocessing (CLAHE) |
| [Pillow](https://pillow.readthedocs.io/) | Image loading & conversion |
| [NumPy](https://numpy.org/) | Array operations & statistics |
| [Requests](https://requests.readthedocs.io/) | URL-based image downloading |

---

## 📁 Project Structure

```
SCRATCH_DETECTION/
│
├── app.py                  # Main Streamlit application
├── best.pt                 # Trained YOLOv11 model weights
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
│
└── archive/                # Dataset
    ├── train_images/       # Training images
    ├── train_annotations/  # Training bounding box labels
    ├── valid_images/       # Validation images
    └── valid_annotations/  # Validation bounding box labels
```

---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/scratch-detection.git
cd scratch-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🧪 How It Works

1. **Input** — User uploads an image or provides a URL
2. **Preprocessing** — CLAHE contrast enhancement is applied (optional toggle) to make faint scratches more visible
3. **Detection** — The YOLOv11 model (`best.pt`) runs inference at 640px resolution
4. **Output** — Annotated image with bounding boxes, scratch count, and confidence scores are displayed

```
Image → CLAHE Enhancement → YOLOv11 Inference → Annotated Output + Metrics
```

---

## ⚙️ Configuration

| Parameter | Range | Default | Description |
|-----------|-------|---------|-------------|
| Confidence Threshold | 0.05 – 0.10 | 0.10 | Lower = more detections, Higher = only obvious scratches |
| Boost Image Contrast | On / Off | On | Applies CLAHE preprocessing |

---

## 📦 Requirements

```
streamlit
ultralytics
Pillow
numpy
requests
opencv-python
```

> Install with: `pip install -r requirements.txt`

---

## 🤖 Model

- **Architecture:** YOLOv11
- **Framework:** Ultralytics
- **Input Size:** 640 × 640 px
- **Trained On:** Custom annotated dataset of metal surface images
- **Weights File:** `best.pt`

---
