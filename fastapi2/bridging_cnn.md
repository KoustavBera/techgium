Visual Disease Classification Pipeline
This document outlines how we will integrate single-frame image classification models (like YOLO or MobileNet) into the existing continuous webcam stream, ensuring a seamless experience for the patient without manual photo uploads.

User Review Required
IMPORTANT

Please review the "Auto-Cropping Pipeline" concept below. It relies on using the existing MediaPipe mesh to find facial regions (like the eyes or cheeks) and cropping them before sending them to the disease classifiers.

Proposed Architecture: The "Auto-Cropping" Engine
Because standard datasets (like JaundEye) consist of closely cropped images of eyes, we cannot feed full webcam frames into the model. We must bridge the gap programmatically.

The Pipeline Flow
Continuous Capture: The webcam runs normally at 30fps.
Snapshot Trigger: Instead of analyzing every frame for diseases (which is slow), we take a single high-quality snapshot during the 60-second vital sign scan.
MediaPipe Landmarking: We run MediaPipe FaceMesh on this snapshot. MediaPipe gives us the exact X,Y pixel coordinates of the left eye, right eye, and face boundaries.
Auto-Cropping (OpenCV): Using the coordinates from MediaPipe, we use OpenCV to strictly crop the image down to just the eye bounding box.
Inference: We feed this small, specific eye crop into our trained Jaundice CNN (or Skin Lesion model).
Biomarker Output: The model outputs a probability (e.g., "Jaundice: 85%"). We add this as a new Biomarker to the BiomarkerSet.
Proposed Changes
app/core/extraction/visual_classification.py
[NEW] VisualDiseaseClassifier
A new Extractor class that inherits from BaseExtractor.
Responsible for taking a raw frame and performing the crop-and-classify logic.
app/hardware/manager.py (or pipeline coordinator)
[MODIFY] Incorporate Visual Scanner
Add the VisualDiseaseClassifier to the list of extractors run during the standard scan cycle.
Verification Plan
Automated Tests
Pass static images of faces into the extractor and verify that the auto-cropping logic correctly isolates the eyes and skin regions before inference.
Use dummy model weights to verify that the extracted score is correctly appended to the output BiomarkerSet.

# Visual Disease Detection with RTX 3060 6GB
## Complete Implementation Guide

---

## 📦 Model Saving & Loading Best Practices

### Framework-Save Formats

| Framework | Save Format | Load Command | Best For |
|-----------|------------|--------------|----------|
| PyTorch (YOLOv8) | `.pt` | `torch.load()` / `YOLO('model.pt')` | Development & Training |
| TensorFlow/Keras | `.h5` or SavedModel | `tf.keras.models.load_model()` | TF Ecosystem |
| TFLite | `.tflite` | `tf.lite.Interpreter()` | Edge Deployment (Kiosk) |
| ONNX | `.onnx` | `onnxruntime.InferenceSession()` | Production (GPU + CPU) |

**💡 Recommendation**: Train in PyTorch → Export to `.pt` for dev, `.onnx` for production deployment

---

## 🎯 Disease-by-Disease Implementation Guide

### 1. 🔴 Skin Lesions (Pox, Rashes, Eczema, Dermatitis)

#### 📊 Datasets
- **HAM10000**: 10,000+ dermoscopy images, 7 classes
- **Skin Disease Dataset**: 23 conditions, 19,500 images (Kaggle)

#### 🚀 Quick Start with API
```python
# Skinive API (Free tier available)
import requests

response = requests.post(
    "https://api.skinive.com/v1/analyze",
    headers={"X-API-Key": "your_key"},
    files={"image": open("lesion.jpg", "rb")}
)
```

#### 🏋️ Training with RTX 3060
```python
from ultralytics import YOLO

# Load pretrained model
model = YOLO("yolov8n-cls.pt")

# Fine-tune (batch_size=32 works with 6GB VRAM)
results = model.train(
    data="path/to/skin_dataset/",
    epochs=50,
    imgsz=224,
    device=0,
    batch=32,
    project="models/skin_disease"
)
```

---

### 2. 🟡 Jaundice (Yellowing of Eyes/Skin)

#### 📊 Dataset
- **JaundEye Dataset**: Eye-cropped jaundice images
- **Icterus Detection 2021**: Research dataset (publicly linked)

#### 🏋️ Training Script
```python
import torch
import torchvision.models as models
from torch.utils.data import DataLoader

# MobileNetV3 for binary classification
model = models.mobilenet_v3_small(pretrained=True)
model.classifier[-1] = torch.nn.Linear(1024, 2)  # 2 classes: Jaundiced/Normal

# Training time: ~15 minutes on RTX 3060
train_loader = DataLoader(dataset, batch_size=64)
# ... training loop
```

---

### 3. 🔴 Anaemia (Pale Conjunctiva)

#### 📊 Dataset
- **Anaemia Detection from Eye Images**: Palpebral conjunctiva dataset

#### 🏋️ Training Script
```python
# Binary CNN - very lightweight
import torch.nn as nn

class AnaemiaCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 54 * 54, 2)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# ~200 images needed for transfer learning
```

---

### 4. 🟠 Obesity / Abdominal Protrusion

#### 🤖 Zero Training Required!
YOLOv8-pose works out of the box with COCO pretrained weights

```python
from ultralytics import YOLO

# Load pretrained pose model
pose_model = YOLO("yolov8n-pose.pt")

# Calculate waist-to-height ratio
results = pose_model(image)
keypoints = results[0].keypoints.xy[0]  # [nose, eyes, shoulders, hips, knees]

# Extract shoulder and hip coordinates
left_shoulder, right_shoulder = keypoints[5], keypoints[6]
left_hip, right_hip = keypoints[11], keypoints[12]

# Calculate ratios
shoulder_width = torch.norm(right_shoulder - left_shoulder)
hip_width = torch.norm(right_hip - left_hip)
ratio = hip_width / shoulder_width  # >1.2 indicates obesity
```

---

### 5. 🟢 Conjunctivitis (Red Eyes)

#### 📊 Dataset
- **Eye Disease Classification**: 4 classes (Normal, Cataract, Glaucoma, Diabetic Retinopathy)

#### 🌐 API Option
```python
# Google Cloud Vision API
from google.cloud import vision

client = vision.ImageAnnotatorClient()
response = client.face_detection(image=image)

# Anomaly detection in eye regions
for face in response.face_annotations:
    # Extract eye landmarks and analyze redness
    pass
```

---

## 🌐 Pre-trained APIs (No Training Needed)

| API | Capabilities | Free Tier |
|-----|-------------|-----------|
| **Google Cloud Vision** | Face detection, skin anomalies | 1000 calls/month |
| **Amazon Rekognition** | Face analysis, body pose | 5000 calls/month |
| **Roboflow** | Host YOLOv8 models | Free public models |
| **Skinive API** | Skin disease classification | Free dev tier |

### Roboflow Quick Integration
```python
import requests

# 2 lines to use pre-trained skin disease model
response = requests.post(
    "https://detect.roboflow.com/skin-disease-model/1",
    params={"api_key": "YOUR_KEY"},
    files={"file": open("image.jpg", "rb")}
)
```

---

## 🏋️ Complete Training Workflow

### Step 1: Environment Setup
```bash
pip install ultralytics torch torchvision onnxruntime
```

### Step 2: Training Script for Multiple Diseases
```python
# train_all_models.py
from ultralytics import YOLO
import torch
import torchvision.models as models

class DiseaseTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using: {self.device}")
    
    def train_skin_lesion(self):
        """Train on HAM10000 - ~2 hours"""
        model = YOLO("yolov8n-cls.pt")
        model.train(
            data="HAM10000/",
            epochs=50,
            imgsz=224,
            device=0,
            batch=32
        )
        model.export(format="onnx")  # Export for production
    
    def train_jaundice(self):
        """Train binary classifier - ~15 mins"""
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(1024, 2)
        # ... training loop
        torch.save(model.state_dict(), "models/jaundice.pt")
    
    def train_anaemia(self):
        """Train binary CNN - ~10 mins"""
        # ... anaemia training code
        pass

if __name__ == "__main__":
    trainer = DiseaseTrainer()
    trainer.train_skin_lesion()  # Priority 1
```

---

## 🎯 Priority Implementation Plan for Techgium

| Priority | Disease | Approach | Time | VRAM Usage |
|----------|---------|----------|------|------------|
| **🔴 P0** | Skin Lesions | Fine-tune YOLOv8n-cls on HAM10000 | 2 hours | 4-5 GB |
| **🔴 P0** | Abdominal Obesity | YOLOv8n-pose (pretrained) | 1 hour | 2 GB |
| **🟠 P1** | Jaundice | Train MobileNetV3 on JaundEye | 1 hour | 2 GB |
| **🟡 P2** | Conjunctivitis | Roboflow API / Binary CNN | 30 min | N/A |
| **🟡 P2** | Anaemia | Binary CNN | 1 hour | 2 GB |

---

## 💻 VisualDiseaseClassifier Implementation

```python
# extractors/visual_disease.py
from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from .base import BaseExtractor
from ..models import BiomarkerSet

class VisualDiseaseClassifier(BaseExtractor):
    def __init__(self):
        """Initialize all models (load once at startup)"""
        print("🔄 Loading visual disease models...")
        
        # Load fine-tuned models (after training)
        self.skin_model = YOLO("models/skin_disease/weights/best.pt")
        self.pose_model = YOLO("yolov8n-pose.pt")
        
        # Load PyTorch models for jaundice/anaemia
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Jaundice model (MobileNetV3)
        self.jaundice_model = torch.load("models/jaundice.pt", map_location=self.device)
        self.jaundice_model.eval()
        
        # Anaemia model (Custom CNN)
        self.anaemia_model = torch.load("models/anaemia.pt", map_location=self.device)
        self.anaemia_model.eval()
        
        # Image transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        print("✅ All visual disease models loaded")
    
    def extract(self, data: dict) -> BiomarkerSet:
        """Main extraction method"""
        biomarker_set = self._create_biomarker_set("visual_disease")
        
        # Get frames from existing pipeline
        face_frame = data.get("raw_face_frames", [None])[0]
        body_frame = data.get("raw_frames", [None])[0]
        
        if face_frame is None:
            return biomarker_set
        
        # 1. Skin Lesion Detection
        skin_results = self.skin_model(face_frame, device=0, verbose=False)[0]
        top_class = skin_results.names[skin_results.probs.top1]
        confidence = float(skin_results.probs.top1conf)
        
        if confidence > 0.5 and top_class != "normal":
            self._add_biomarker(
                biomarker_set,
                "skin_condition",
                value=confidence,
                description=f"Visual: {top_class} ({confidence:.0%} confidence)"
            )
        
        # 2. Jaundice Detection (from eye region)
        # Use MediaPipe landmarks to crop eye region
        face_landmarks = data.get("face_landmarks")
        if face_landmarks:
            # Extract eye region logic here
            eye_img = self._extract_eye_region(face_frame, face_landmarks)
            
            # Classify
            with torch.no_grad():
                eye_tensor = self.transform(Image.fromarray(eye_img)).unsqueeze(0).to(self.device)
                output = self.jaundice_model(eye_tensor)
                pred = torch.softmax(output, dim=1)
                jaundice_prob = pred[0][1].item()  # Class 1 = Jaundiced
                
                if jaundice_prob > 0.6:
                    self._add_biomarker(
                        biomarker_set,
                        "jaundice_flag",
                        value=jaundice_prob,
                        description=f"Jaundice probability: {jaundice_prob:.0%}"
                    )
        
        # 3. Obesity Detection (body frame)
        if body_frame is not None:
            pose_results = self.pose_model(body_frame, device=0, verbose=False)[0]
            
            if pose_results.keypoints is not None:
                keypoints = pose_results.keypoints.xy[0]
                
                # Calculate waist-to-height ratio
                if len(keypoints) > 12:  # Has hip keypoints
                    left_hip, right_hip = keypoints[11], keypoints[12]
                    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
                    
                    hip_width = torch.norm(right_hip - left_hip).item()
                    shoulder_width = torch.norm(right_shoulder - left_shoulder).item()
                    
                    ratio = hip_width / shoulder_width
                    
                    self._add_biomarker(
                        biomarker_set,
                        "obesity_indicator",
                        value=ratio,
                        description=f"Waist-to-shoulder ratio: {ratio:.2f}"
                    )
        
        return biomarker_set
    
    def _extract_eye_region(self, frame, landmarks):
        """Extract eye region using face landmarks"""
        # Implement eye cropping logic
        h, w = frame.shape[:2]
        
        # Get eye landmarks indices (MediaPipe)
        left_eye_idx = [33, 133, 157, 158, 159, 160, 161, 173]
        right_eye_idx = [362, 263, 387, 386, 385, 384, 398, 466]
        
        # Calculate bounding box for eyes
        all_eye_points = []
        for idx in left_eye_idx + right_eye_idx:
            landmark = landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            all_eye_points.append([x, y])
        
        all_eye_points = np.array(all_eye_points)
        x_min, y_min = all_eye_points.min(axis=0)
        x_max, y_max = all_eye_points.max(axis=0)
        
        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        return frame[y_min:y_max, x_min:x_max]
```

---

## 📝 Quick Start Guide

### 1. Install Dependencies
```bash
pip install ultralytics torch torchvision opencv-python pillow numpy
```

### 2. Train Priority Models
```python
# train_priority.py
from train_all_models import DiseaseTrainer

trainer = DiseaseTrainer()
trainer.train_skin_lesion()  # 2 hours
# trainer.train_jaundice()   # 1 hour (next)
```

### 3. Integrate Extractor
```python
# In your main pipeline
from extractors.visual_disease import VisualDiseaseClassifier

visual_extractor = VisualDiseaseClassifier()

# During processing
biomarkers = visual_extractor.extract({
    "raw_face_frames": [face_image],
    "raw_frames": [body_image],
    "face_landmarks": landmarks
})
```

---

## 🔧 Performance Optimization for RTX 3060

### Memory Management
```python
# Clear cache between inferences
torch.cuda.empty_cache()

# Use mixed precision training
with torch.cuda.amp.autocast():
    outputs = model(images)
```

### Batch Size Recommendations
| Model | Max Batch Size | VRAM |
|-------|---------------|------|
| YOLOv8n-cls | 64 | 4 GB |
| YOLOv8n-pose | 32 | 3 GB |
| MobileNetV3 | 128 | 2 GB |
| Custom CNN | 256 | 2 GB |

---

## 🚀 Production Deployment

### Export to ONNX
```python
# Export for production
model = YOLO("models/skin_disease/weights/best.pt")
model.export(format="onnx", imgsz=224)

# Inference with ONNX Runtime
import onnxruntime as ort
sess = ort.InferenceSession("models/skin_disease/weights/best.onnx")
results = sess.run(None, {"images": input_tensor})
```

### Docker Configuration
```dockerfile
FROM pytorch/pytorch:latest
RUN pip install ultralytics onnxruntime-gpu
COPY models/ /app/models/
COPY extractors/ /app/extractors/
CMD ["python", "-m", "main"]
```

---

## 📊 Expected Results

| Disease | Accuracy | Inference Time | Model Size |
|---------|----------|----------------|------------|
| Skin Lesions | 85-90% | 15ms | 12 MB |
| Jaundice | 92-95% | 5ms | 8 MB |
| Anaemia | 88-92% | 3ms | 4 MB |
| Obesity | N/A | 20ms | 6 MB |

---

This guide provides everything needed to implement visual disease detection with your RTX 3060. Start with P0 items and progressively add more biomarkers!