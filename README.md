# CIFAR-10 Image Classification with Transfer Learning

A computer vision project comparing a baseline CNN trained from scratch against a MobileNetV2 transfer learning model on the CIFAR-10 dataset.

## Results

| Model | Test Accuracy |
|-------|--------------|
| Baseline CNN (from scratch) | 75.64% |
| MobileNetV2 (transfer learning) | **80.76%** |

Transfer learning improved accuracy by **~5%** while using significantly fewer trainable parameters.

## Dataset

**CIFAR-10** — 60,000 RGB images across 10 classes (50,000 train / 10,000 test)

Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Project Structure
```
├── code.ipynb        # Main notebook with all code and outputs
└── README.md
```

## Approach

### 1. Data Preprocessing
- Normalized pixel values from [0, 255] to [0, 1]
- Applied data augmentation: random rotation, horizontal flip, zoom
- Resized images to 96×96 for MobileNetV2 compatibility

### 2. Baseline CNN
Built a simple CNN from scratch:
- 3 × (Conv2D → MaxPooling2D) blocks
- Flatten → Dense(256) → Dropout(0.5) → Dense(10, softmax)
- Trained with Adam optimizer and categorical crossentropy loss

### 3. Transfer Learning Model (MobileNetV2)
- Loaded MobileNetV2 pretrained on ImageNet (`include_top=False`)
- Froze all base layers (2,257,984 frozen params) to preserve learned features
- Added custom head: GlobalAveragePooling2D → Dense(128) → Dropout(0.3) → Dense(10, softmax)
- Only 165,258 parameters trained on CIFAR-10

### 4. Training
- Used `EarlyStopping` (patience=5, monitor=val_accuracy) to prevent overfitting
- Max epochs set to 50; training stopped automatically at optimal point

### 5. Evaluation
- Compared accuracy on 10,000 test images
- Generated confusion matrices and classification reports for both models

## Key Findings

- MobileNetV2 achieved **80.76% accuracy** vs **75.64%** for the baseline CNN
- Transfer learning converged faster — reaching ~76% validation accuracy in epoch 1 alone
- Freezing pretrained layers preserved ImageNet features (edges, textures, shapes) and prevented catastrophic forgetting
- EarlyStopping ensured the best weights were restored automatically

## Tech Stack

- Python 3.11
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- scikit-learn

## Concepts Demonstrated

- Transfer learning and feature reuse
- Catastrophic forgetting and why layer freezing matters
- Data augmentation to reduce overfitting
- Model evaluation with confusion matrices and classification reports
- Early stopping for optimal training
