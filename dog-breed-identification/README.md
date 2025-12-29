# Dog Breed Identification 🐕

An end-to-end deep learning project for multi-class dog breed classification using TensorFlow 2.0 and Transfer Learning with MobileNetV2.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Visualization](#visualization)
- [Kaggle Submission](#kaggle-submission)
- [Improvements](#improvements)
- [References](#references)

## 🎯 Overview

This project implements a deep learning model that can accurately classify images of dogs into their respective breeds. It uses transfer learning with a pre-trained MobileNetV2 model from TensorFlow Hub to identify 120 different dog breeds from the Stanford Dogs Dataset.

**Problem**: Given an image of a dog, predict its breed from 120 possible classes.

**Solution**: Transfer learning using MobileNetV2 pre-trained on ImageNet, fine-tuned for dog breed classification.

**Evaluation**: Multi-class log loss (Kaggle competition metric)

> **Note**: This project is part of the [Complete Machine Learning & Data Science Bootcamp 2025](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/) on Udemy - a comprehensive zero-to-mastery course covering machine learning, deep learning, and data science.

## ✨ Features

- **Transfer Learning**: Uses pre-trained MobileNetV2 from TensorFlow Hub
- **Modular Architecture**: Clean OOP design with separate classes for data, model, training, and prediction
- **GPU Support**: Optimized for GPU training with TensorFlow
- **Visualization**: Built-in functions for visualizing predictions and confidence scores
- **Model Persistence**: Save and load trained models
- **Kaggle Ready**: Generate submission files in Kaggle format
- **TensorBoard Integration**: Monitor training progress in real-time
- **Early Stopping**: Prevent overfitting with automatic early stopping

## 📊 Dataset

The project uses the **Stanford Dogs Dataset** from Kaggle's Dog Breed Identification competition.

- **Training Images**: 10,222 labeled images
- **Test Images**: 10,357 unlabeled images
- **Classes**: 120 different dog breeds
- **Image Format**: JPEG
- **Average Images per Breed**: ~85 images

### Download Dataset

1. Visit [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/competitions/dog-breed-identification)
2. Download the dataset files:
   - `train.zip` - Training images
   - `test.zip` - Test images
   - `labels.csv` - Training labels

3. Extract the files:
   ```bash
   unzip train.zip
   unzip test.zip
   ```

Your directory structure should look like:
```
dog-breed-identification/
├── train/          # Training images
├── test/           # Test images
└── labels.csv      # Training labels
```

## 🔧 Requirements

### System Requirements
- **Python**: 3.11+ (tested with Python 3.11)
- **OS**: macOS (tested on Mac OS with M4 Pro chip), Linux, or Windows
- **RAM**: 8GB+ (16GB recommended)
- **Disk Space**: 5GB+
- **GPU**:
  - **Apple Silicon (M1/M2/M3/M4)**: Uses `tensorflow-metal` for GPU acceleration
  - **NVIDIA GPU**: CUDA-enabled GPU (optional but recommended)
  - **CPU**: Supported but slower

### Tested Configuration
✅ This project has been **tested and verified** to work on:
- **Hardware**: Mac with M4 Pro chip
- **OS**: macOS Tahoe (26.1.0)
- **Python**: 3.11
- **TensorFlow**: 2.16.2 with Metal acceleration (`tensorflow-metal==1.1.0`)

The `requirements.txt` file contains all package versions that are confirmed to work on **Apple Silicon Macs (M1/M2/M3/M4)**.

## 📥 Installation

> **📌 For Mac M4 Pro Users**: These instructions and the included `requirements.txt` have been specifically tested and verified to work on Mac M4 Pro with macOS Tahoe (26.1.0). The setup includes `tensorflow-metal` for GPU acceleration on Apple Silicon.

### Step 0: Verify Python Version

Ensure you have Python 3.11 installed:

```bash
python3 --version
```

Expected output:
```
Python 3.11.x
```

If you don't have Python 3.11, install it:
- **macOS**: `brew install python@3.11` (requires Homebrew)
- **Windows**: Download from [python.org](https://www.python.org/downloads/)
- **Linux**: Use your package manager (e.g., `sudo apt install python3.11`)

### Step 1: Clone or Navigate to Project Directory

```bash
cd /path/to/dog-breed-identification
```

### Step 2: Create Virtual Environment

#### On macOS/Linux:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Verify activation (should show path to .venv)
which python
```

#### On Windows:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Verify activation
where python
```

### Step 3: Upgrade pip

```bash
python -m pip install --upgrade pip
```

### Step 4: Install Dependencies from requirements.txt

The project includes a `requirements.txt` file with all tested package versions that work on **Apple Silicon Macs (M1/M2/M3/M4)**.

```bash
# Install all dependencies at once
pip install -r requirements.txt
```

This will install:
- ✅ `tensorflow==2.16.2` - Deep learning framework
- ✅ `tensorflow-metal==1.1.0` - GPU acceleration for Apple Silicon (M1/M2/M3/M4)
- ✅ `tensorflow-hub==0.16.1` - Pre-trained models
- ✅ `tf_keras==2.16.0` - Keras compatibility layer
- ✅ `numpy`, `pandas`, `matplotlib`, `scikit-learn` - Data science libraries
- ✅ `jupyter`, `ipywidgets` - Notebook environment
- ✅ And all other dependencies (~140 packages)

**Note**: The installation may take 5-10 minutes depending on your internet speed.

### Step 5: Verify Installation

```bash
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__); print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

Expected output on **Mac with M4 Pro**:
```
TensorFlow version: 2.16.2
GPU Available: True
```

You should see:
- ✅ `GPU Available: True` on Apple Silicon Macs (M1/M2/M3/M4) - Metal GPU acceleration
- ✅ `GPU Available: True` on systems with NVIDIA CUDA GPU
- ℹ️ `GPU Available: False` on CPU-only systems (still works, just slower)

To see detailed GPU info on Apple Silicon:
```bash
python -c "import tensorflow as tf; print('GPUs:', tf.config.list_physical_devices('GPU'))"
```

Expected output on M4 Pro:
```
GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Deactivating Virtual Environment

When you're done working:

```bash
deactivate
```

## 📁 Project Structure

```
dog-breed-identification/
│
├── train/                          # Training images directory
│   ├── 000bec180eb18c.jpg
│   ├── 001513dfcb2ffaf.jpg
│   └── ...
│
├── test/                           # Test images directory
│   ├── 266490a2b053a4a.jpg
│   └── ...
│
├── my-dog-photos/                  # Custom images for testing
│   └── ...
│
├── models/                         # Saved models directory
│   ├── 20251229-001234-experiment1.h5
│   └── ...
│
├── logs/                           # TensorBoard logs
│   ├── 20251229-001234/
│   └── ...
│
├── labels.csv                      # Training labels
├── dog_vision.py                   # Main Python module (OOP version)
├── dog-vision.ipynb                # Jupyter notebook (original)
├── dog_vision_mindmap.md           # System architecture mind map
├── README.md                       # This file
├── requirements.txt                # Python dependencies
└── dog_breed_predictions_kaggle_format.csv  # Kaggle submission file
```

## 🚀 Usage

### Quick Start

```python
from dog_vision import DogBreedClassifier

# Initialize classifier
classifier = DogBreedClassifier()

# Prepare data (using subset for faster training)
classifier.prepare_data(num_images=4000, test_size=0.2)

# Build model
classifier.build_model()

# Train model
classifier.train(epochs=10, use_early_stopping=True)

# Evaluate on validation set
loss, accuracy = classifier.evaluate()
print(f"Validation Accuracy: {accuracy:.2%}")

# Save model
model_path = classifier.save(suffix="experiment-1")

# Create Kaggle submission
classifier.create_submission()
```

### Training on Full Dataset

```python
from dog_vision import DogBreedClassifier

# Initialize classifier
classifier = DogBreedClassifier()

# Use all training data
classifier.prepare_data()  # No num_images parameter = use all

# Build and train
classifier.build_model()
classifier.train(epochs=100, use_early_stopping=True)

# Save
classifier.save(suffix="full-dataset-mobilenetv2")
```

### Loading and Using a Saved Model

```python
from dog_vision import DogBreedClassifier

# Initialize classifier
classifier = DogBreedClassifier()

# Prepare data (needed for unique_breeds)
classifier.prepare_data(num_images=4000)

# Load saved model
classifier.load("models/20251229-001234-experiment1.h5")

# Evaluate
classifier.evaluate()

# Create submission
classifier.create_submission()
```

### Custom Predictions

```python
from dog_vision import DogBreedClassifier

# Initialize and load model
classifier = DogBreedClassifier()
classifier.prepare_data()
classifier.load("models/your-model.h5")

# Predict on custom images
test_data, test_filenames = classifier.prepare_test_data("my-dog-photos/")
predictions = classifier.predict(test_data)

# Visualize
from matplotlib import pyplot as plt
images, labels = classifier.predictor.unbatchify(test_data)
classifier.predictor.plot_prediction(predictions, labels, images, n=0)
plt.show()
```

### Running the Jupyter Notebook

```bash
# Make sure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Launch Jupyter
jupyter notebook

# Open dog-vision.ipynb in your browser
```

## 🏗️ Model Architecture

### Transfer Learning with MobileNetV2

```
Input (224x224x3)
      ↓
MobileNetV2 (Pre-trained on ImageNet)
├── 5,432,713 parameters (frozen)
├── Trained on 1000 ImageNet classes
└── Feature extraction layer
      ↓
Dense Layer (120 units, softmax)
├── 120,240 trainable parameters
└── Output: Dog breed probabilities
      ↓
Output (120 classes)
```

### Model Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | MobileNetV2 (TensorFlow Hub) |
| Input Size | 224x224x3 |
| Output Classes | 120 |
| Total Parameters | 5,552,953 |
| Trainable Parameters | 120,240 |
| Non-trainable Parameters | 5,432,713 |
| Loss Function | Categorical Crossentropy |
| Optimizer | Adam |
| Batch Size | 32 |

## 📈 Results

### Training Performance

**Training on 4,000 images (10 epochs):**

| Metric | Training | Validation |
|--------|----------|------------|
| Accuracy | ~99% | ~78% |
| Loss | ~0.03 | ~0.72 |

**Training on Full Dataset (10,222 images):**

| Metric | Training |
|--------|----------|
| Accuracy | >99% |
| Loss | <0.02 |

### Notes on Performance

- High training accuracy with lower validation accuracy indicates overfitting
- This is expected with transfer learning and limited data augmentation
- The model learns very quickly due to pre-trained weights
- Early stopping helps prevent excessive overfitting

## 📊 Visualization

### Training Metrics (TensorBoard)

View training progress in real-time:

```bash
# Make sure you're in the project directory
tensorboard --logdir logs/

# Open browser to http://localhost:6006
```

### Prediction Visualizations

The project includes several visualization functions:

1. **Single Prediction Plot**: Shows image with predicted breed, confidence, and true label (color-coded)
2. **Top-K Predictions**: Bar chart showing top 10 most likely breeds
3. **Image Grid**: 5x5 grid of predictions for quick overview

```python
# Example: Visualize predictions
from matplotlib import pyplot as plt

# Get predictions
predictions = classifier.predict(classifier.val_data)

# Unbatch validation data
images, labels = classifier.predictor.unbatchify(classifier.val_data)

# Plot single prediction
classifier.predictor.plot_prediction(predictions, labels, images, n=0)
plt.show()

# Plot confidence chart
classifier.predictor.plot_prediction_confidence(predictions, labels, n=0, top_k=10)
plt.show()

# Show grid of 25 images
classifier.predictor.show_images_grid(images[:25], labels[:25], grid_size=5)
plt.show()
```

## 🏆 Kaggle Submission

### Generate Submission File

```python
# After training your model
classifier.create_submission()

# This creates: dog_breed_predictions_kaggle_format.csv
```

### Submission File Format

The CSV file contains:
- `id` column: Image filename (without .jpg extension)
- 120 breed columns: Probability for each breed (sum to 1.0)

### Submit to Kaggle

1. Go to [Kaggle Competition Page](https://www.kaggle.com/competitions/dog-breed-identification)
2. Click "Submit Predictions"
3. Upload `dog_breed_predictions_kaggle_format.csv`
4. View your score (Multi-Class Log Loss)

## 🔄 Improvements

### Ways to Improve Model Performance

1. **Data Augmentation**
   ```python
   # Add to preprocessing
   - Random horizontal flips
   - Random rotations (±15°)
   - Random crops and resizing
   - Color jittering
   - Random brightness/contrast
   ```

2. **Fine-tuning**
   ```python
   # Unfreeze top layers of MobileNetV2
   # Train with lower learning rate
   base_model.trainable = True
   # Freeze early layers, train later ones
   ```

3. **Different Architectures**
   - EfficientNetB0-B7
   - ResNet50/101/152
   - InceptionV3
   - Vision Transformer (ViT)

4. **Ensemble Methods**
   - Train multiple models
   - Average predictions
   - Weighted voting

5. **Hyperparameter Tuning**
   - Learning rate scheduling
   - Different optimizers (AdamW, SGD with momentum)
   - Batch size experiments
   - Regularization (dropout, L2)

6. **More Training Data**
   - Use full 10,222 training images
   - External dog breed datasets
   - Synthetic data generation

## 🐛 Troubleshooting

### Common Issues

**1. GPU Not Detected**
```bash
# Check CUDA installation
nvidia-smi

# Install GPU version of TensorFlow
pip install tensorflow[and-cuda]
```

**2. Out of Memory Error**
```python
# Reduce batch size
classifier = DogBreedClassifier(batch_size=16)  # or 8
```

**3. Module Not Found Error**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**4. Slow Training**
- Use GPU if available
- Reduce number of images for testing
- Reduce batch size if GPU memory limited

**5. TensorFlow Hub Download Issues**
```bash
# Set cache directory
export TFHUB_CACHE_DIR=/path/to/cache

# Or use different model URL
```

## 📚 References

### Course
- [Complete Machine Learning & Data Science Bootcamp 2025](https://www.udemy.com/course/complete-machine-learning-and-data-science-zero-to-mastery/) - Zero to Mastery course on Udemy

### Dataset & Competition
- [Kaggle Dog Breed Identification Competition](https://www.kaggle.com/competitions/dog-breed-identification)
- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/)

### Model & Documentation
- [TensorFlow Hub - MobileNetV2](https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5)
- [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)
- [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)
- [TensorFlow Metal Plugin for Apple Silicon](https://developer.apple.com/metal/tensorflow-plugin/)

## 📝 License

This project is for educational purposes. Dataset is from Kaggle's Dog Breed Identification competition.

## 👤 Author

**Sohail**

- GitHub: [@sohail](https://github.com/sohail)
- Project: Dog Breed Identification using Deep Learning

## 🙏 Acknowledgments

- Kaggle for providing the dataset and competition platform
- TensorFlow team for the excellent framework and pre-trained models
- Stanford University for the Stanford Dogs Dataset
- The deep learning community for tutorials and resources

---

## 🚦 Quick Command Reference

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Train
python -c "from dog_vision import DogBreedClassifier; c = DogBreedClassifier(); c.prepare_data(4000); c.build_model(); c.train(10)"

# Jupyter
jupyter notebook dog-vision.ipynb

# TensorBoard
tensorboard --logdir logs/

# Deactivate
deactivate
```

---

**Happy Training! 🐕🎉**

For more detailed architecture information, see `dog_vision_mindmap.md`
