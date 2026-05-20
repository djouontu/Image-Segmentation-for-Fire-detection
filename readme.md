# Fire Detection using Mask R-CNN: Complete Project Documentation

---

## Introduction and Use Case

### What This Project Does
This project detects and segments fire regions in images using computer vision and deep learning. It can:
- **Detect** where fires are located in images (bounding boxes)
- **Segment** the exact shape of fire regions (pixel-level masks)
- **Work in real-time** for fire monitoring systems
- **Handle various fire scenarios** (indoor fires, wildfires, etc.)

### Real-World Applications
- **Forest Fire Monitoring**: Automated detection of wildfires from satellite or drone imagery
- **Industrial Safety**: Monitor factories, warehouses for fire hazards
- **Home Security**: Smart surveillance systems for fire detection
- **Emergency Response**: Help firefighters quickly locate fire regions
- **Insurance Assessment**: Automated damage assessment from fire incidents

---

## What is Mask R-CNN?

### The Basics (For Beginners)
Imagine you're looking at a photo and want to:
1. **Find all objects** in the image (like finding all people in a crowd photo)
2. **Draw boxes around them** (object detection)
3. **Trace their exact outline** (instance segmentation)

That's exactly what Mask R-CNN does, but instead of people, we're finding fire regions!

### Technical Explanation
**Mask R-CNN** (Region-based Convolutional Neural Network with Mask prediction) is a deep learning model that performs three tasks simultaneously:

1. **Object Detection**: Finds objects and puts bounding boxes around them
2. **Classification**: Identifies what each object is (in our case: "fire" or "background")
3. **Instance Segmentation**: Creates pixel-perfect masks showing the exact shape of each object

### How Mask R-CNN Works (Step by Step)

```
Input Image → Feature Extraction → Region Proposal → Classification + Box Regression + Mask Generation → Output
```

1. **Feature Extraction**: Uses a backbone network (ResNet-50) to extract features from the image
2. **Region Proposal Network (RPN)**: Suggests potential object locations
3. **ROI Align**: Extracts features for each proposed region
4. **Classification Head**: Determines what each region contains (fire vs background)
5. **Box Regression Head**: Refines the bounding box coordinates
6. **Mask Head**: Generates pixel-level masks for each detected object

---

## Solution Approach

### Why PyTorch?
1. **Modern and Stable**: PyTorch has excellent backward compatibility
2. **Built-in Mask R-CNN**: `torchvision` provides ready-to-use Mask R-CNN models
3. **Active Maintenance**: Regular updates and community support
4. **Simpler API**: More intuitive for beginners
5. **Python 3.12+ Compatible**: No version conflicts

### Implementation Strategy
1. **Use Pre-trained Models**: Start with COCO-trained Mask R-CNN
2. **Transfer Learning**: Fine-tune for fire detection
3. **Custom Dataset Class**: Handle VIA annotation format
4. **Modern Training Loop**: Use PyTorch best practices
5. **Comprehensive Evaluation**: Test on validation and new images

---

## Technical Implementation

### Model Architecture
```python
# Load pre-trained Mask R-CNN with ResNet-50 backbone
model = maskrcnn_resnet50_fpn(pretrained=True)

# Customize for our use case (2 classes: background + fire)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Update mask predictor
in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)
```

### Key Components

#### 1. **Backbone Network**: ResNet-50 + Feature Pyramid Network (FPN)
- **ResNet-50**: Extracts hierarchical features from images
- **FPN**: Combines features from different scales for better detection

#### 2. **Region Proposal Network (RPN)**
- Generates potential object locations
- Uses anchor boxes at multiple scales and aspect ratios

#### 3. **ROI Heads**
- **Box Predictor**: Classifies regions and refines bounding boxes
- **Mask Predictor**: Generates segmentation masks

### Training Configuration
```python
NUM_CLASSES = 2        # background + fire
BATCH_SIZE = 2         # Small due to memory constraints
NUM_EPOCHS = 10        # Quick training for demonstration
LEARNING_RATE = 0.005  # Standard learning rate for fine-tuning
MOMENTUM = 0.9         # SGD momentum
WEIGHT_DECAY = 0.0005  # L2 regularization
```

---

## Dataset and Annotations

### Dataset Structure
```
input/dataset/
├── train/
│   ├── 1.jpg, 28.jpg, 49.jpg, ...     # Fire images
│   └── via_project.json               # Annotations
├── val/
│   ├── validation images
│   └── via_project.json
└── test/
    └── test images
```

### Annotation Format: VIA (VGG Image Annotator)
VIA is a popular tool for image annotation that creates JSON files with polyline annotations.

#### Sample Annotation Structure:
```json
{
  "1.jpg20356": {
    "filename": "1.jpg",
    "size": 20356,
    "regions": [
      {
        "shape_attributes": {
          "name": "polyline",
          "all_points_x": [173, 201, 202, 235, ...],
          "all_points_y": [257, 249, 207, 203, ...]
        },
        "region_attributes": {
          "fire": "fire"
        }
      }
    ]
  }
}
```


---


## Code Structure Explanation

### 1. Data Loading (`FireDataset` Class)

#### Purpose
Convert VIA annotations into PyTorch-compatible format.

#### Key Methods
```python
class FireDataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        # Load VIA JSON annotations
        # Filter images with annotations
        
    def __getitem__(self, idx):
        # Load image
        # Extract polygon coordinates  
        # Create binary masks using cv2.fillPoly()
        # Generate bounding boxes
        # Return image and targets
```

#### Data Flow
```
VIA JSON → Polygon Coordinates → Binary Masks → Bounding Boxes → PyTorch Tensors
```

### 2. Model Creation (`get_model` Function)

#### Purpose
Create and customize Mask R-CNN for fire detection.

#### Process
```python
def get_model(num_classes):
    # 1. Load pre-trained model
    model = maskrcnn_resnet50_fpn(pretrained=True)
    
    # 2. Replace classifier head for our classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 3. Replace mask predictor for our classes
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    return model
```

### 3. Training Loop (`train_one_epoch` Function)

#### Process Flow
```python
for epoch in range(num_epochs):
    for batch in dataloader:
        # 1. Forward pass
        loss_dict = model(images, targets)
        
        # 2. Compute total loss
        losses = sum(loss for loss in loss_dict.values())
        
        # 3. Backward pass
        losses.backward()
        
        # 4. Update weights
        optimizer.step()
```

### 4. Inference (`predict_and_visualize` Function)

#### Purpose
Run trained model on new images and visualize results.

#### Process
```python
model.eval()
with torch.no_grad():
    predictions = model([image])
    
# Extract results
boxes = predictions[0]['boxes']
scores = predictions[0]['scores']  
masks = predictions[0]['masks']

# Filter by confidence threshold
keep = scores >= threshold
```

---
# Local Execution Instructions

## Python version 3.12

To create a virtual environment and install requirements in Python 3.12 on different operating systems, follow the instructions below:

### For Windows:

Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.

Change the directory to the desired location for your project:

```sh
cd C:\path\to\project
```

Create a new virtual environment using the `venv` module:

```sh
python -m venv myenv
```

Activate the virtual environment:

```sh
myenv\Scripts\activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

```sh
cd /path/to/project
```

Create a new virtual environment using the `venv` module:

```sh
python3.12 -m venv myenv
```

Activate the virtual environment:

```sh
source myenv/bin/activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

These instructions assume you have Python 3.12 installed and added to your system's `PATH` variable.

## Execution Instructions if Multiple Python Versions Installed

If you have multiple Python versions installed on your system, you can use the Python Launcher to create a virtual environment with Python 3.12. Specify the version using the `-p` or `--python` flag. Follow the instructions below:

### For Windows:

Open the Command Prompt by pressing `Win + R`, typing `cmd`, and pressing `Enter`.

Change the directory to the desired location for your project:

```sh
cd C:\path\to\project
```

Create a new virtual environment using the Python Launcher:

```sh
py -3.12 -m venv myenv
```

> **Note**: Replace `myenv` with your desired virtual environment name.

Activate the virtual environment:

```sh
myenv\Scripts\activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

### For Linux/Mac:

Open a terminal.

Change the directory to the desired location for your project:

```sh
cd /path/to/project
```

Create a new virtual environment using the Python Launcher:

```sh
python3.12 -m venv myenv
```

> **Note**: Replace `myenv` with your desired virtual environment name.

Activate the virtual environment:

```sh
source myenv/bin/activate
```

Install the project requirements using pip:

```sh
pip install -r requirements.txt
```

By specifying the version using `py -3.12` or `python3.12`, you can ensure that the virtual environment is created using Python 3.12 specifically, even if you have other Python versions installed.



## How to Use This Project



### Step-by-Step Usage

#### 1. **Prepare Your Dataset**
```
input/dataset/
├── train/
│   ├── fire_image1.jpg
│   ├── fire_image2.jpg
│   └── via_project.json
```

#### 2. **Run Training**
# Load notebook: FireMask-PyTorch.ipynb



#### 3. **Inference on New Images**
```python
# Load trained model
model = load_model_for_inference(MODEL_PATH, NUM_CLASSES, device)

# Predict on new image
predictions = predict_and_visualize(model, test_image_path)
```

### Configuration Options
```python
# Modify these in cell 3:
NUM_CLASSES = 2        # background + fire
BATCH_SIZE = 2         # Adjust based on GPU memory
NUM_EPOCHS = 10        # More epochs for better results
LEARNING_RATE = 0.005  # Learning rate for fine-tuning
```

---

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. **"Number of objects: 0" Error**
**Cause**: Annotation format mismatch
```python
# Check your VIA JSON format:
# Look for "name": "polygon" or "name": "polyline"
# Update FireDataset accordingly
```

#### 2. **CUDA Out of Memory**
**Solution**: Reduce batch size
```python
BATCH_SIZE = 1  # Reduce from 2 to 1
```

#### 3. **ImportError for torchvision**
**Solution**: Install correct versions
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. **Poor Detection Results**
**Causes and Solutions**:
- **Insufficient Training Data**: Add more annotated images
- **Low Image Quality**: Use higher resolution images
- **Incorrect Annotations**: Verify polygon accuracy
- **Wrong Confidence Threshold**: Lower the threshold (0.3 instead of 0.5)

