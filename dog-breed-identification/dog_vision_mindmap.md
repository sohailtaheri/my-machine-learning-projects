# Dog Breed Classification - Mind Map

## Overview
```
                                    🐕 DOG BREED CLASSIFIER 🐕
                                              |
                    ┌─────────────────────────┼─────────────────────────┐
                    |                         |                         |
            📊 DATA PIPELINE          🧠 MODEL PIPELINE          📈 PREDICTION PIPELINE
```

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DogBreedClassifier (Main Orchestrator)                 │
│                                                                                   │
│  Main Methods:                                                                    │
│  • prepare_data() ──────────► Prepare train/val splits                          │
│  • build_model() ───────────► Create model architecture                          │
│  • train() ─────────────────► Train the model                                    │
│  • evaluate() ──────────────► Evaluate on validation set                         │
│  • predict() ───────────────► Make predictions                                   │
│  • save() / load() ─────────► Model persistence                                  │
│  • create_submission() ─────► Generate Kaggle submission                         │
└─────────────────────────────────────────────────────────────────────────────────┘
                    |                    |                    |
        ┌───────────┘                    |                    └───────────┐
        │                                |                                │
        ▼                                ▼                                ▼
```

---

## 📊 Component 1: Data Processing

```
┌──────────────────────────────────────────────────────────────┐
│              📦 DataPreprocessor                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Initialization:                                              │
│  ├─ labels_csv_path: "labels.csv"                           │
│  ├─ train_dir: "train/"                                      │
│  ├─ img_size: 224                                            │
│  └─ batch_size: 32                                           │
│                                                               │
│  Core Methods:                                                │
│  ├─ load_labels()                                            │
│  │   ├─ Read CSV file                                        │
│  │   ├─ Create file paths                                    │
│  │   ├─ Extract unique breeds (120 classes)                  │
│  │   └─ Create boolean label matrix                          │
│  │                                                            │
│  ├─ process_image(image_path)                                │
│  │   ├─ tf.io.read_file() ──► Read image                    │
│  │   ├─ tf.image.decode_jpeg() ──► Decode                   │
│  │   ├─ convert_image_dtype() ──► Normalize [0,1]           │
│  │   └─ tf.image.resize() ──► Resize to 224x224             │
│  │                                                            │
│  ├─ get_image_label(image_path, label)                       │
│  │   └─ Return (image_tensor, label) tuple                   │
│  │                                                            │
│  ├─ create_data_batches(X, y, valid_data, test_data)        │
│  │   ├─ Training: Shuffle + Batch                           │
│  │   ├─ Validation: No shuffle + Batch                      │
│  │   └─ Test: No labels + Batch                             │
│  │                                                            │
│  └─ prepare_train_val_split(num_images, test_size)          │
│      ├─ Load all data                                        │
│      ├─ Train/Val split (80/20)                             │
│      └─ Create batched datasets                             │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
                    TensorFlow Datasets
                  (Batched & Preprocessed)
```

---

## 🧠 Component 2: Model Building & Training

```
┌──────────────────────────────────────────────────────────────┐
│                  🏗️ ModelBuilder                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Configuration:                                               │
│  ├─ img_size: 224                                            │
│  └─ model_url: TensorFlow Hub MobileNetV2                    │
│                                                               │
│  create_model(num_classes):                                  │
│  └─ Architecture:                                            │
│      ┌────────────────────────────────────┐                 │
│      │ Input Layer (224x224x3)            │                 │
│      └────────────┬───────────────────────┘                 │
│                   ▼                                          │
│      ┌────────────────────────────────────┐                 │
│      │ TF Hub: MobileNetV2 (Frozen)       │                 │
│      │ - Pre-trained on ImageNet          │                 │
│      │ - 5.4M non-trainable params        │                 │
│      └────────────┬───────────────────────┘                 │
│                   ▼                                          │
│      ┌────────────────────────────────────┐                 │
│      │ Dense Layer (120 units)            │                 │
│      │ - Activation: Softmax              │                 │
│      │ - 120K trainable params            │                 │
│      └────────────┬───────────────────────┘                 │
│                   ▼                                          │
│      ┌────────────────────────────────────┐                 │
│      │ Output: Breed Probabilities        │                 │
│      └────────────────────────────────────┘                 │
│                                                               │
│  Compilation:                                                 │
│  ├─ Loss: CategoricalCrossentropy                           │
│  ├─ Optimizer: Adam                                          │
│  └─ Metrics: Accuracy                                        │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  🎯 ModelTrainer                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Callbacks:                                                   │
│  ├─ create_tensorboard_callback()                           │
│  │   ├─ Create log directory                                │
│  │   ├─ Timestamp: YYYYMMDD-HHMMSS                          │
│  │   └─ Monitor training metrics                            │
│  │                                                            │
│  └─ create_early_stopping_callback()                        │
│      ├─ Monitor: val_accuracy                                │
│      └─ Patience: 3 epochs                                   │
│                                                               │
│  train_model():                                              │
│  ├─ Setup callbacks                                          │
│  ├─ Fit model on training data                              │
│  ├─ Validate each epoch                                      │
│  └─ Return trained model                                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                  💾 ModelManager                              │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  save_model(model, suffix):                                  │
│  ├─ Create models directory                                  │
│  ├─ Generate filename with timestamp                         │
│  └─ Save as .h5 file                                         │
│                                                               │
│  load_model(model_path):                                     │
│  ├─ Load from .h5 file                                       │
│  └─ Register custom KerasLayer                              │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 📈 Component 3: Prediction & Visualization

```
┌──────────────────────────────────────────────────────────────┐
│                  🔮 Predictor                                 │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Core Functions:                                              │
│                                                               │
│  ├─ get_pred_label(prediction_probabilities)                │
│  │   └─ Return breed name from argmax                        │
│  │                                                            │
│  ├─ unbatchify(data)                                         │
│  │   ├─ Extract images from batches                          │
│  │   └─ Extract labels from batches                          │
│  │                                                            │
│  ├─ plot_prediction(probs, labels, images, n)               │
│  │   ├─ Display image                                        │
│  │   ├─ Show predicted breed                                 │
│  │   ├─ Show confidence %                                     │
│  │   └─ Color: Green (correct) / Red (wrong)                │
│  │                                                            │
│  ├─ plot_prediction_confidence(probs, labels, n, top_k)     │
│  │   ├─ Get top K predictions                                │
│  │   ├─ Create bar chart                                     │
│  │   └─ Highlight correct breed in green                     │
│  │                                                            │
│  ├─ show_images_grid(images, labels, grid_size)             │
│  │   ├─ Create 5x5 grid                                      │
│  │   ├─ Display 25 images                                    │
│  │   └─ Show breed labels                                    │
│  │                                                            │
│  └─ create_kaggle_submission(predictions, filenames)        │
│      ├─ Create DataFrame with image IDs                      │
│      ├─ Add probability columns for 120 breeds               │
│      └─ Save as CSV for Kaggle submission                    │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

---

## 🔄 Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING WORKFLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

1️⃣ DATA PREPARATION
   │
   ├─► Load labels.csv (10,222 images)
   ├─► Extract 120 unique dog breeds
   ├─► Create boolean label matrix (10222 x 120)
   ├─► Split into train/validation (80/20)
   └─► Create TensorFlow batched datasets

2️⃣ IMAGE PREPROCESSING
   │
   ├─► Read image files from disk
   ├─► Decode JPEG to tensor
   ├─► Normalize pixel values [0, 1]
   ├─► Resize to 224x224x3
   └─► Batch into groups of 32

3️⃣ MODEL BUILDING
   │
   ├─► Load MobileNetV2 from TensorFlow Hub
   ├─► Freeze pre-trained layers (5.4M params)
   ├─► Add Dense output layer (120 classes)
   └─► Compile with Adam optimizer

4️⃣ TRAINING
   │
   ├─► Setup TensorBoard logging
   ├─► Setup Early Stopping (patience=3)
   ├─► Train for N epochs
   ├─► Validate after each epoch
   └─► Monitor val_accuracy

5️⃣ EVALUATION
   │
   ├─► Compute validation loss
   ├─► Compute validation accuracy
   └─► Visualize predictions

6️⃣ SAVING
   │
   └─► Save model to models/TIMESTAMP-suffix.h5

┌─────────────────────────────────────────────────────────────────────────────┐
│                          PREDICTION WORKFLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

1️⃣ LOAD MODEL
   │
   └─► Load saved .h5 model

2️⃣ PREPARE TEST DATA
   │
   ├─► Load test images
   ├─► Preprocess (normalize, resize)
   └─► Create batches

3️⃣ PREDICT
   │
   ├─► Forward pass through model
   └─► Get probability distribution (120 classes)

4️⃣ VISUALIZATION
   │
   ├─► Plot predictions with confidence
   ├─► Show top-K predictions
   └─► Display image grids

5️⃣ SUBMISSION
   │
   └─► Create CSV with probabilities for all breeds
```

---

## 📋 Key Parameters & Configurations

```
┌────────────────────────────────────────────────────────────┐
│                    HYPERPARAMETERS                          │
├────────────────────────────────────────────────────────────┤
│                                                             │
│  Data:                                                      │
│  ├─ Total Images: 10,222                                   │
│  ├─ Number of Classes: 120 dog breeds                      │
│  ├─ Train/Val Split: 80/20                                 │
│  └─ Batch Size: 32                                          │
│                                                             │
│  Image Processing:                                          │
│  ├─ Input Size: 224 x 224 x 3                             │
│  ├─ Color Space: RGB                                        │
│  └─ Normalization: [0, 1]                                   │
│                                                             │
│  Model:                                                     │
│  ├─ Base: MobileNetV2 (ImageNet pre-trained)              │
│  ├─ Total Params: ~5.5M                                    │
│  ├─ Trainable Params: 120K                                 │
│  └─ Non-trainable Params: 5.4M                             │
│                                                             │
│  Training:                                                  │
│  ├─ Loss: Categorical Crossentropy                         │
│  ├─ Optimizer: Adam                                         │
│  ├─ Metrics: Accuracy                                       │
│  ├─ Epochs: 10 (default)                                    │
│  └─ Early Stopping: Patience 3                             │
│                                                             │
└────────────────────────────────────────────────────────────┘
```

---

## 🎯 Use Cases

```
┌─────────────────────────────────────────────────────────────┐
│                    USAGE SCENARIOS                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1️⃣ Quick Training (Subset):                                │
│     classifier.prepare_data(num_images=4000)                │
│     classifier.build_model()                                │
│     classifier.train(epochs=10)                             │
│                                                              │
│  2️⃣ Full Training (All Data):                               │
│     classifier.prepare_data()  # Uses all 10K images        │
│     classifier.build_model()                                │
│     classifier.train(epochs=100)                            │
│                                                              │
│  3️⃣ Transfer Learning:                                       │
│     # Model uses pre-trained MobileNetV2                    │
│     # Only trains final classification layer                │
│                                                              │
│  4️⃣ Model Persistence:                                       │
│     model_path = classifier.save("experiment-1")            │
│     classifier.load(model_path)                             │
│                                                              │
│  5️⃣ Kaggle Competition:                                      │
│     classifier.create_submission()                          │
│     # Generates: dog_breed_predictions_kaggle_format.csv    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔍 Data Flow Diagram

```
labels.csv ─────┐
                │
train/*.jpg ────┼──► DataPreprocessor ──► TF Dataset (Batched)
                │                              │
                │                              │
                └──────────────────────────────┤
                                               │
                                               ▼
                                        ModelBuilder
                                               │
                                               ▼
                                      MobileNetV2 Model
                                               │
                                               ▼
                                        ModelTrainer
                                       (with callbacks)
                                               │
                                               ▼
                                       Trained Model ──┬──► ModelManager ──► .h5 file
                                               │       │
                                               │       └──► Predictor ──► Visualizations
                                               │                      └──► Kaggle CSV
                                               ▼
                                        Predictions
                                      (120 probabilities)
```

---

## 🚀 Quick Start Example

```python
# 1. Import
from dog_vision import DogBreedClassifier

# 2. Initialize
classifier = DogBreedClassifier()

# 3. Prepare Data
classifier.prepare_data(num_images=4000, test_size=0.2)

# 4. Build Model
classifier.build_model()

# 5. Train
classifier.train(epochs=10, use_early_stopping=True)

# 6. Evaluate
loss, accuracy = classifier.evaluate()

# 7. Save
model_path = classifier.save(suffix="my-experiment")

# 8. Make Predictions on Test Set
classifier.create_submission()
```

---

## 🎨 Visualization Outputs

```
┌──────────────────────────────────────────────────────────┐
│           VISUALIZATION CAPABILITIES                      │
├──────────────────────────────────────────────────────────┤
│                                                           │
│  📊 Training Metrics (TensorBoard):                      │
│  ├─ Loss curves (train/val)                             │
│  ├─ Accuracy curves (train/val)                          │
│  └─ Real-time monitoring                                 │
│                                                           │
│  🖼️ Prediction Plots:                                    │
│  ├─ Single prediction with image                         │
│  ├─ Confidence percentage                                │
│  └─ Correct/Incorrect color coding                       │
│                                                           │
│  📈 Confidence Charts:                                   │
│  ├─ Top-K predictions bar chart                          │
│  └─ True label highlighted                               │
│                                                           │
│  🎞️ Image Grids:                                         │
│  └─ 5x5 grid of images with labels                       │
│                                                           │
└──────────────────────────────────────────────────────────┘
```

---

## 🏆 Model Performance Notes

```
Expected Performance (from notebook):
├─ Training on 4,000 images (10 epochs):
│  ├─ Training Accuracy: ~99%
│  └─ Validation Accuracy: ~78%
│
├─ Training on all 10,222 images:
│  └─ Training Accuracy: >99%
│
└─ Note: High training accuracy with lower validation
         indicates overfitting - normal for transfer learning
         with limited data augmentation
```

---

## 🔧 Improvement Strategies

```
📈 Ways to Improve Model Performance:

1️⃣ Data Augmentation
   ├─ Random flips
   ├─ Random rotations
   ├─ Random crops
   └─ Color jittering

2️⃣ Fine-tuning
   ├─ Unfreeze top layers of MobileNetV2
   └─ Train with lower learning rate

3️⃣ Different Architectures
   ├─ EfficientNet
   ├─ ResNet
   └─ Inception

4️⃣ Ensemble Methods
   └─ Combine predictions from multiple models

5️⃣ More Data
   └─ Use full 10K+ training set
```
