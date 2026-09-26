# Concealed - Feature-Targeted Image Processor

An intelligent image processing and privacy-preserving tool that detects specific visual features (face, arms, text, tables, water bottles, laptops, and arbitrary objects) and applies modification algorithms strictly conforming to the object's silhouette.

## Quickstart for Team Members

### 1. Clone the Repository
```bash
git clone https://github.com/AshtonLittle/Concealed.git
cd Concealed
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Note on Model Weights**: Model weights (`*.pt` and `*.onnx`) are deliberately excluded from Git. They are automatically downloaded on your first run:
> - **YOLO11m-seg**, **YOLO11m-pose**, **YOLOv8m-worldv2**, and **FastSAM-s** are downloaded automatically by Ultralytics.
> - **PP-OCRv3** text detector is downloaded automatically by the script.

### 3. Run the Processor

```bash
# Modify faces conforming to head contour
python image_processor.py my_photo.png --feature face

# Modify arms conforming to limbs
python image_processor.py my_photo.png --feature arms

# Modify all text in the image
python image_processor.py document.png --feature text

# Modify any custom or arbitrary object
python image_processor.py room.png --feature laptop
python image_processor.py room.png --feature "brick wall"
python image_processor.py street.png --feature people

# Multiple features at once
python image_processor.py my_photo.png --feature "face, laptop"
```

### Customizing the Algorithm
Open [`image_processor.py`](image_processor.py) and place your logic inside `apply_algorithm(section, mask)`.