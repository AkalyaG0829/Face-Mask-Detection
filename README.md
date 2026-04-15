# Face Mask Detection

A real-time face mask detection system using deep learning and computer vision. This project uses a Convolutional Neural Network (CNN) built with TensorFlow/Keras and OpenCV's DNN-based face detector to identify whether people are wearing masks or not through a webcam feed.

## Features

- **Real-time Detection**: Process webcam frames in real-time to detect masks
- **Pre-trained Model**: Uses a trained MobileNetV2-based CNN classifier
- **Face Localization**: OpenCV DNN face detector for accurate face localization
- **Visual Feedback**: Color-coded bounding boxes (green for mask, red for no mask)
- **Confidence Scoring**: Displays prediction confidence for each detection

## Project Structure

```
face_mask_detection/
├── detect_mask.py                          # Real-time detection script
├── train_model.py                          # Model training script
├── mask_detector.h5                        # Trained mask detection model
├── deploy.prototxt.txt                     # OpenCV face detector config
├── res10_300x300_ssd_iter_140000.caffemodel  # OpenCV face detector weights
├── requirements.txt                        # Python dependencies
└── dataset/
    ├── with_mask/                          # Training images with masks
    └── without_mask/                       # Training images without masks
```

## Requirements

- Python 3.7+
- TensorFlow >= 2.10.0
- OpenCV >= 4.7.0
- NumPy >= 1.23.0
- scikit-learn >= 1.2.0
- Matplotlib >= 3.7.0
- Pillow >= 9.4.0

## Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training the Model

If you want to retrain the model with your own dataset:

```bash
python train_model.py
```

**Dataset Structure Required**:
```
dataset/
├── with_mask/      # Images of people wearing masks
└── without_mask/   # Images of people without masks
```

**Dataset Source**: You can download a dataset from [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset)

**Configuration** (in `train_model.py`):
- `IMAGE_SIZE`: Input size for the model (224x224 for MobileNetV2)
- `BATCH_SIZE`: Batch size for training (default: 32)
- `EPOCHS`: Number of training epochs (default: 20)
- `LEARNING_RATE`: Learning rate (default: 1e-4)
- `TEST_SPLIT`: Percentage of data for testing (default: 20%)

### Running Detection

For real-time mask detection via webcam:

```bash
python detect_mask.py
```

**Controls**:
- Press `q` to quit the application

**Configuration** (in `detect_mask.py`):
- `CONFIDENCE_THRESHOLD`: Minimum confidence to consider a detection valid (default: 0.5)
- Color scheme can be customized via `COLOR_MASK`, `COLOR_NO_MASK`, and `COLOR_TEXT`

## Model Architecture

The detection model is built on:
- **Base Model**: MobileNetV2 (pre-trained on ImageNet)
- **Top Layers**:
  - Average Pooling 2D
  - Flatten
  - Dense (128 units, ReLU activation)
  - Dropout (0.5)
  - Dense (2 units, Softmax activation)
- **Optimizer**: Adam
- **Loss**: Binary Crossentropy

## Face Detection

Uses OpenCV's DNN module with:
- **Detector**: Single-Shot MultiBox Detector (SSD)
- **Framework**: Caffe
- **Input Size**: 300x300
- **Source**: [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector)

## Performance

Results depend on your dataset quality and training configuration. Typical performance:
- Accuracy: 95%+ on test set
- Inference Speed: Real-time on CPU (~30-50 FPS depending on hardware)
- Latency: <50ms per frame on modern systems

## Limitations

- Works best in adequate lighting conditions
- Accuracy depends on training dataset diversity
- Subjects must be facing the camera for reliable detection
- Performance varies based on mask type and coverage

## Future Improvements

- Support for multiple face mask types
- GPU acceleration for faster processing
- REST API endpoint for integration
- Web-based interface
- Mobile app deployment
- Improved lighting-invariant detection

## Troubleshooting

**No faces detected?**
- Check camera permissions
- Ensure adequate lighting
- Verify face models are in the correct directory

**Low accuracy?**
- Retrain the model with more diverse data
- Increase number of epochs
- Adjust hyperparameters in `train_model.py`

**Model file not found?**
- Ensure `mask_detector.h5` is in the project directory
- Train the model first: `python train_model.py`

## License

MIT License - See LICENSE file for details

## Author

Akalya.G

