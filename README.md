# License Plate Recognition and Tracking 🚗


This project detects vehicles in a video, tracks them, identifies their license plates, and uses OCR to read the license plate numbers. It produces a final video with annotated bounding boxes and license plate information.

---
## Features ✨

- **Vehicle Detection**: Uses the YOLOv8 model to detect cars, trucks, and buses.
- **Object Tracking**: Employs the SORT algorithm to assign a unique ID to each detected vehicle and track it across frames.
- **License Plate Detection**: A custom-trained YOLO model pinpoints the location of license plates.
- **Optical Character Recognition (OCR)**: Uses EasyOCR to read the characters from the detected license plates.
- **Data Interpolation**: Fills in missing frames where a vehicle might not have been detected to ensure smooth tracking.

---
## Project Structure

```
license-plate-recognition/
│
├── data/                  # Input/Output data
├── models/                # Trained model weights
├── src/                   # Source code
├── .gitignore             # Files for Git to ignore
├── README.md              # You are here!
└── requirements.txt       # Project dependencies
```

---
## Installation ⚙️

Follow these steps to set up the project environment.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/license-plate-recognition.git](https://github.com/YOUR_USERNAME/license-plate-recognition.git)
    cd license-plate-recognition
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Models & Data:**
    * Place your input video in the `data/videos/` directory (e.g., `data/videos/temp.mp4`).
    * Ensure the model files (`license_plate_detector.pt` and `yolov8n.pt`) are in the `models/` directory.

---
## How to Run 🚀

The process is broken down into three main steps. Run the scripts from the `src/` directory.

### Step 1: Run Detection and Tracking
This script processes the input video, detects and tracks vehicles, performs OCR on license plates, and saves the raw data to `test.csv`.

```bash
python main.py
```
* **Input**: `data/videos/temp.mp4`
* **Output**: `data/output/test.csv`

### Step 2: Interpolate Missing Data
This script reads the raw CSV, fills in any frames where tracking was temporarily lost, and creates a new, complete CSV file.

```bash
python add_missing_data.py
```
* **Input**: `data/output/test.csv`
* **Output**: `data/output/test_interpolated.csv`

### Step 3: Visualize the Results
This final script reads the interpolated data and the original video to create an output video with all the tracking and license plate annotations.

```bash
python visualize.py
```
* **Input**: `data/output/test_interpolated.csv` and `data/videos/temp.mp4`
* **Output**: `data/output/out.mp4`

You will find the final annotated video at `data/output/out.mp4`. Enjoy! 🎉
