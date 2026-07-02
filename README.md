# 🚗 YOLOv8 Vehicle Detection, Tracking and Counting
## Overview
A real-time vehicle detection, tracking, and counting system built with YOLOv8 and ByteTrack. The system detects multiple vehicle classes, tracks them with persistent IDs, and accurately counts each vehicle as it crosses a virtual counting line.
## 🎬 Demo
<p align="center">
  <img src="count_car.gif" width="900">
</p>
# 📂 Dataset

This project uses a custom vehicle detection dataset obtained from **Roboflow Universe**.

## Download Dataset

1. Visit **Roboflow Universe**.
2. Search for a vehicle detection dataset (e.g., cars, trucks, buses, motorcycles).
3. Click **Download Dataset**.
4. Select the **YOLOv8** export format.
5. Extract the downloaded dataset into your project directory.

Example project structure:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml

#  Counting Logic

1. Detect vehicles using YOLOv8.
2. Track each vehicle with ByteTrack.
3. Assign a unique ID to every object.
4. Compute the center point of each vehicle.
5. Count the vehicle only once after its center crosses the virtual counting line.
6. Display live statistics on the screen.

#  Requirements

* Python 3.8+
* OpenCV
* NumPy
* Ultralytics YOLOv8
* PyTorch

# Installation

## 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository.git

cd your-repository
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install opencv-python numpy ultralytics torch
```

or

```bash
pip install -r requirements.txt
```

---

# Model

Place your trained YOLOv8 model inside the project folder.
```text
best.pt
```
If you do not have a custom model, you can use one of the official YOLOv8 pretrained models by renaming it to **best.pt**.

---

# 🎥 Input Video

Place your traffic video in the project directory.

Example:

```text
demo.mp4
```

Or modify the last line in `check.py`:

```python
detector.run("your_video.mp4")
```



#  Configuration

Several parameters can be adjusted inside `check.py`.

| Parameter        | Description                    |
| ---------------- | ------------------------------ |
| `line_y`         | Position of the counting line  |
| `conf`           | Detection confidence threshold |
| `iou`            | IoU threshold for NMS          |
| `target_classes` | Vehicle categories to count    |

Example:

```python
self.line_y = 550

results = self.model.track(
    frame,
    conf=0.5,
    iou=0.4,
    persist=True
)
```

### Cannot open video

Make sure:

* The video exists.
* The file path is correct.
* OpenCV supports the video codec.

---

### ModuleNotFoundError

Install the missing package.

```bash
pip install ultralytics
```

---

### No vehicle detected

Possible causes:

* Incorrect model
* Confidence threshold too high
* Unsupported vehicle class

Try lowering:

```python
conf=0.3
```

---

# 📌 Future Improvements

* Vehicle speed estimation
* Lane detection
* Traffic density analysis
* Vehicle direction detection
* Dashboard with analytics
* Export statistics to CSV
* Web application with Streamlit
  
# License

This project is intended for educational and research purposes.

# 👨‍💻 Author

Le Nhat Truong
