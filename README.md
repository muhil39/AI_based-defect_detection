# AI-Based Surface Defect Detection in Manufactured Components

An industrial-grade, locally hosted AI inspection web application designed for mechanical engineering quality assurance. The system leverages laptop webcam imaging and sequential Convolutional Neural Networks (CNN) to first identify component material and then conduct high-precision surface crack detection on metal components.

---

## 1. Project Overview & Architecture

### Sequential Inspection Logic Pipeline
```
[User Positions Component]
          │
          ▼
[Click "Capture & Inspect"] ──▶ (Webcam frame frozen & saved)
          │
          ▼
[AI Stage 1: Material Classification CNN]
          │
  ┌───────┴───────────────────────────────┐
  │                                       │
  ▼                                       ▼
[MATERIAL = METAL]              [MATERIAL = PLASTIC / OTHER]
  │                                       │
  ▼                                       ▼
[AI Stage 2: Crack Detection CNN]   [Skip Crack Inspection]
  │                                       │
  ├──▶ CRACK DETECTED                     ├──▶ "Crack inspection omitted"
  │      - Status: REJECT                 └──▶ Status: SKIPPED
  │      - Audio Warning Activated
  └──▶ NO CRACK DETECTED
         - Status: ACCEPTABLE
          │
          ▼
[Store Record in Local SQLite Database] ──▶ [Update Live Dashboard & Logs]
```

1. **Optical Capture Stage**: Images are captured **strictly on demand** when the operator clicks the **Capture & Inspect** button. The live webcam feed is not continuously analyzed, avoiding compute overload.
2. **Material Classification (CNN 1)**: Categorizes the component into **METAL**, **PLASTIC**, or **OTHER / UNKNOWN**. If the classification confidence falls below 55%, the system safely outputs `OTHER / UNKNOWN` rather than forcing an inaccurate classification.
3. **Surface Crack Detection (CNN 2)**: Gated strictly to components confirmed as **METAL**. The CNN detects fracture lines and surface fissures, reporting **CRACK DETECTED** or **NO VISIBLE CRACK DETECTED**.
4. **Voice Alert System**: When a crack is detected on metal, an immediate audible voice warning is synthesized through the laptop speakers via the browser's Web Speech API: *"Warning! Crack detected on the metal surface."*
5. **Quality Audit Logging**: All inspection records, timestamps, confidence scores, and image references are stored locally in an embedded SQLite database (`inspections.db`).

---

## 2. Technology Stack

- **Frontend**: HTML5, CSS3 (Modern Industrial Dark Theme, Cyan & Deep Navy Accents), Vanilla JavaScript (Zero external CDN dependencies).
- **Audio Synthesis**: Browser `SpeechSynthesis` Web Speech API.
- **Backend**: Python 3.10+ / 3.12, Flask 3.x RESTful API.
- **Deep Learning**: TensorFlow / Keras (Convolutional Neural Networks, Transfer Learning).
- **Computer Vision & Image Processing**: OpenCV (`opencv-python-headless`), Pillow (`PIL`), NumPy.
- **Database**: SQLite3 (Local file-based, zero configuration).
- **Cost**: 100% Free and Open Source. No cloud APIs, no subscriptions.

---

## 3. Project Directory Structure

```
AI_Surface_Defect_Detection/
├── app.py                     # Main Flask server and REST API controller
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation & execution guide
├── inspections.db             # Local SQLite inspection audit database
├── backend/
│   ├── __init__.py            # Backend package initialization
│   ├── database.py            # SQLite database schema, inserts, history & stats
│   ├── image_processing.py    # Base64 decoding, OpenCV resizing, normalization
│   ├── material_model.py      # Material classification CNN inference wrapper
│   └── crack_model.py         # Metal crack detection CNN inference wrapper
├── models/
│   ├── material_classifier.keras  # Trained Material CNN model
│   └── crack_detector.keras       # Trained Crack Detection CNN model
├── frontend/
│   ├── index.html             # Industrial SCADA-themed dashboard interface
│   ├── style.css              # Cyber-industrial styling with HUD crosshairs & badges
│   └── script.js              # Webcam controller, audio speech, and UI sync
├── uploads/                   # Storage for captured inspection images
├── dataset/                   # Dataset directory for training
│   ├── material/
│   │   ├── metal/             # Metal component images
│   │   ├── plastic/           # Plastic component images
│   │   └── other/             # Other non-metallic objects
│   └── cracks/
│       ├── crack/             # Metal surfaces containing cracks
│       └── no_crack/          # Clean, nominal metal surfaces
└── training/
    ├── generate_sample_data.py # Procedural benchmark texture generator
    ├── train_material.py       # CNN training pipeline for Material Classifier
    └── train_crack.py          # CNN training pipeline for Crack Detector
```

---

## 4. Setup & Installation Instructions

### Step 1: Open Terminal in Project Directory
Navigate to the project root directory:
```powershell
cd C:\Users\muhil\.gemini\antigravity\scratch\AI_Surface_Defect_Detection
```

### Step 2: Install Python Dependencies
Install the required packages using pip:
```powershell
pip install -r requirements.txt
```
*(Dependencies: Flask, opencv-python-headless, Pillow, numpy, tensorflow)*

---

## 5. Dataset Preparation & CNN Model Training

The application expects two trained models in the `models/` directory:
1. `models/material_classifier.keras`
2. `models/crack_detector.keras`

### Option A: Immediate Benchmark Training (Quick Start)
To allow immediate end-to-end testing without downloading large datasets, a procedural benchmark dataset generator is provided:

1. **Generate Benchmark Samples**:
   ```powershell
   python training/generate_sample_data.py
   ```
   *Generates 180 realistic material images and 120 crack/no-crack micro-fracture metal images.*

2. **Train Material Classifier CNN**:
   ```powershell
   python training/train_material.py --epochs 15
   ```
   *Trains the CNN and automatically saves `models/material_classifier.keras`.*

3. **Train Surface Crack Detector CNN**:
   ```powershell
   python training/train_crack.py --epochs 15
   ```
   *Trains the CNN and automatically saves `models/crack_detector.keras`.*

### Option B: Training with Public Datasets (Academic Evaluation)
For final engineering presentations or research papers, you can use standard public academic datasets:

1. **NEU Surface Defect Database (Northeastern University)**:
   - Contains 6 classes of hot-rolled steel strip surface defects: *Cracked, Inclusions, Patches, Pitted Surface, Rolled-in Scale, Scratches*.
   - Place cracked metal images into `dataset/cracks/crack/` and clean surface patches into `dataset/cracks/no_crack/`.
2. **Kaggle Surface Crack Detection Dataset**:
   - 40,000 images divided into Positive (cracked) and Negative (clean).
   - Place positive samples in `dataset/cracks/crack/` and negative in `dataset/cracks/no_crack/`.
3. **Material Identification Dataset**:
   - Collect or photograph manufactured scrap pieces (bolts, washers, aluminum billets, plastic brackets, acrylic sheets, cardboard/wooden blocks).
   - Store in `dataset/material/metal/`, `dataset/material/plastic/`, and `dataset/material/other/`.
4. Re-run `train_material.py` and `train_crack.py` to retrain with your custom datasets.

---

## 6. Running the Web Application

### Launch Local Server
Execute:
```powershell
python app.py
```

The Flask server will start locally:
```
=================================================================
AI-BASED SURFACE DEFECT DETECTION IN MANUFACTURED COMPONENTS
=================================================================
Starting Flask local server on http://127.0.0.1:5000 ...
 * Serving Flask app 'app'
 * Running on http://127.0.0.1:5000
```

### Access Dashboard
Open your web browser (Chrome, Edge, or Firefox) and navigate to:
```
http://127.0.0.1:5000
```

---

## 7. Operational Workflow (Operator Guide)

1. **Start Camera**:
   - Click **Start Camera**.
   - When prompted by the browser, click **Allow** for webcam access.
   - The live feed will display within the HUD target reticle.
2. **Position Component**:
   - Align the manufactured component within the target crosshairs.
   - Ensure adequate lighting on the metallic surface.
3. **Capture & Inspect**:
   - Click the prominent **Capture & Inspect** button.
   - The system freezes the image frame and sends it to the backend.
   - The 5-stage progress stepper illustrates the sequential execution:
     `Capture ➔ Material AI ➔ Metal Gate ➔ Crack CNN ➔ Quality Decision`.
4. **Evaluate Decision**:
   - **Metal Component**: The Material card displays `METAL`. The Crack Detection card evaluates surface integrity.
     - If crack detected: Display turns **RED**, status shows **CRACK DETECTED - REJECT**, and laptop speakers announce *"Warning! Crack detected on the metal surface."*
     - If surface clean: Display turns **GREEN**, status shows **SURFACE NOMINAL - PASS**.
   - **Plastic Component**: The Material card displays `PLASTIC`. Crack detection is **skipped** with the note *"Plastic component detected. Crack inspection is not performed."*
   - **Other / Low Confidence**: Crack detection is **skipped** with the note *"Material not recognised. Inspection cannot be performed."*
5. **Inspection History**:
   - Click the **History & Logs** tab in the top navigation.
   - View past inspections, timestamps, confidence values, and high-resolution captured thumbnails.
   - Click any thumbnail to expand high-resolution inspection modal.
   - Delete individual records or click **Clear All Records** to reset SQLite storage.
6. **Testing Without Webcam**:
   - If testing without a live camera, use the **Browse Image** file selector under the camera controls to upload any component image from disk.

---

## 8. Academic Notes & Engineering Limitations

- **Optical Measurement Notice**: A standard 2D laptop webcam captures color and intensity contrast. It **cannot** measure absolute physical crack depth (in millimeters) or internal volumetric fractures. The CNN detects surface-breaking linear discontinuities based on learned spatial feature representations.
- **Lighting & Glare**: Specular glare from highly polished metals can be mitigated by adjusting lighting angle or using diffuse illumination.
- **Safety First**: Audio warnings ensure quality inspectors on a factory line are alerted immediately without needing their eyes glued to the screen.

