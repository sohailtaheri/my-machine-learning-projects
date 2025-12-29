"""
Dog Breed Classification using Transfer Learning with MobileNetV2

This module implements an end-to-end multi-class image classifier for dog breed identification
using TensorFlow 2.0 and TensorFlow Hub.
"""

import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tf_keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional


class DataPreprocessor:
    """Handles data loading, preprocessing, and batching for dog breed classification."""

    def __init__(self, labels_csv_path: str = "labels.csv",
                 train_dir: str = "train/",
                 img_size: int = 224,
                 batch_size: int = 32):
        """
        Initialize the data preprocessor.

        Args:
            labels_csv_path: Path to the labels CSV file
            train_dir: Directory containing training images
            img_size: Target image size for resizing
            batch_size: Batch size for data batching
        """
        self.labels_csv_path = labels_csv_path
        self.train_dir = train_dir
        self.img_size = img_size
        self.batch_size = batch_size
        self.unique_breeds = None

    def load_labels(self) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """
        Load labels from CSV and create boolean label arrays.

        Returns:
            Tuple of (filenames, boolean_labels, unique_breeds)
        """
        labels_csv = pd.read_csv(self.labels_csv_path)

        # Create filenames
        filenames = [self.train_dir + fname + ".jpg" for fname in labels_csv['id'].values]

        # Get unique breeds
        self.unique_breeds = np.unique(labels_csv['breed'])

        # Create boolean labels
        labels = labels_csv['breed']
        boolean_labels = np.array([labels == breed for breed in self.unique_breeds]).T

        return filenames, boolean_labels, self.unique_breeds

    def process_image(self, image_path: str) -> tf.Tensor:
        """
        Process a single image: read, decode, normalize, and resize.

        Args:
            image_path: Path to the image file

        Returns:
            Processed image tensor
        """
        # Read in the image
        image = tf.io.read_file(image_path)
        # Decode the image into a tensor with 3 color channels (RGB)
        image = tf.image.decode_jpeg(image, channels=3)
        # Convert to float32 (0-1 range)
        image = tf.image.convert_image_dtype(image, tf.float32)
        # Resize the image
        image = tf.image.resize(image, [self.img_size, self.img_size])
        return image

    def get_image_label(self, image_path: str, label: np.ndarray) -> Tuple[tf.Tensor, np.ndarray]:
        """
        Create a tuple of (image, label).

        Args:
            image_path: Path to the image file
            label: Label array

        Returns:
            Tuple of (image_tensor, label)
        """
        image = self.process_image(image_path)
        return image, label

    def create_data_batches(self, X: List[str], y: Optional[np.ndarray] = None,
                           valid_data: bool = False, test_data: bool = False) -> tf.data.Dataset:
        """
        Create batches of data from image paths and labels.

        Args:
            X: List of image file paths
            y: Label arrays (None for test data)
            valid_data: Whether this is validation data (no shuffle)
            test_data: Whether this is test data (no labels)

        Returns:
            Batched TensorFlow dataset
        """
        if test_data:
            print("Creating test data batches...")
            data = tf.data.Dataset.from_tensor_slices(tf.constant(X))
            data_batch = data.map(self.process_image).batch(self.batch_size)
            return data_batch

        elif valid_data:
            print("Creating validation data batches...")
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
            dataset = dataset.map(self.get_image_label)
            dataset = dataset.batch(self.batch_size)
            return dataset

        else:
            print("Creating training data batches...")
            dataset = tf.data.Dataset.from_tensor_slices((tf.constant(X), tf.constant(y)))
            dataset = dataset.shuffle(buffer_size=len(X))
            dataset = dataset.map(self.get_image_label)
            dataset = dataset.batch(self.batch_size)
            return dataset

    def prepare_train_val_split(self, num_images: Optional[int] = None,
                                test_size: float = 0.2,
                                random_state: int = 42) -> Tuple:
        """
        Prepare training and validation data splits.

        Args:
            num_images: Number of images to use (None for all)
            test_size: Proportion of data for validation
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_data, val_data, X_train, X_val, y_train, y_val)
        """
        filenames, boolean_labels, unique_breeds = self.load_labels()

        if num_images:
            filenames = filenames[:num_images]
            boolean_labels = boolean_labels[:num_images]

        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            filenames, boolean_labels, test_size=test_size, random_state=random_state
        )

        # Create batches
        train_data = self.create_data_batches(X_train, y_train)
        val_data = self.create_data_batches(X_val, y_val, valid_data=True)

        return train_data, val_data, X_train, X_val, y_train, y_val


class ModelBuilder:
    """Handles model creation using TensorFlow Hub."""

    def __init__(self, img_size: int = 224,
                 model_url: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"):
        """
        Initialize the model builder.

        Args:
            img_size: Input image size
            model_url: TensorFlow Hub model URL
        """
        self.img_size = img_size
        self.model_url = model_url

    def create_model(self, num_classes: int) -> tf_keras.Model:
        """
        Create a Keras model using transfer learning.

        Args:
            num_classes: Number of output classes

        Returns:
            Compiled Keras model
        """
        print(f"Building model with: {self.model_url}")

        # Setup model layers
        model = tf_keras.Sequential([
            hub.KerasLayer(self.model_url, trainable=False),  # Input layer
            tf_keras.layers.Dense(units=num_classes, activation="softmax")  # Output layer
        ])

        # Compile the model
        model.compile(
            loss=tf_keras.losses.CategoricalCrossentropy(),
            optimizer=tf_keras.optimizers.legacy.Adam(),
            metrics=["accuracy"]
        )

        # Build the model
        input_shape = [None, self.img_size, self.img_size, 3]
        model.build(input_shape)

        return model


class ModelTrainer:
    """Handles model training with callbacks."""

    @staticmethod
    def create_tensorboard_callback() -> tf_keras.callbacks.TensorBoard:
        """
        Create a TensorBoard callback for logging.

        Returns:
            TensorBoard callback
        """
        logdir = os.path.join(
            os.getcwd(),
            "logs",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        os.makedirs(logdir, exist_ok=True)
        return tf_keras.callbacks.TensorBoard(logdir)

    @staticmethod
    def create_early_stopping_callback(monitor: str = "val_accuracy",
                                      patience: int = 3) -> tf_keras.callbacks.EarlyStopping:
        """
        Create an early stopping callback.

        Args:
            monitor: Metric to monitor
            patience: Number of epochs with no improvement before stopping

        Returns:
            EarlyStopping callback
        """
        return tf_keras.callbacks.EarlyStopping(monitor=monitor, patience=patience)

    def train_model(self, model: tf_keras.Model,
                   train_data: tf.data.Dataset,
                   val_data: Optional[tf.data.Dataset] = None,
                   epochs: int = 10,
                   use_tensorboard: bool = True,
                   use_early_stopping: bool = True) -> tf_keras.Model:
        """
        Train the model with callbacks.

        Args:
            model: Keras model to train
            train_data: Training dataset
            val_data: Validation dataset (optional)
            epochs: Number of training epochs
            use_tensorboard: Whether to use TensorBoard callback
            use_early_stopping: Whether to use early stopping callback

        Returns:
            Trained model
        """
        callbacks = []

        if use_tensorboard:
            callbacks.append(self.create_tensorboard_callback())

        if use_early_stopping:
            monitor = "val_accuracy" if val_data else "accuracy"
            callbacks.append(self.create_early_stopping_callback(monitor=monitor))

        # Train the model
        model.fit(
            x=train_data,
            epochs=epochs,
            validation_data=val_data,
            validation_freq=1 if val_data else None,
            callbacks=callbacks
        )

        return model


class ModelManager:
    """Handles model saving and loading."""

    @staticmethod
    def save_model(model: tf_keras.Model, suffix: Optional[str] = None) -> str:
        """
        Save a trained model.

        Args:
            model: Keras model to save
            suffix: Optional suffix for the filename

        Returns:
            Path to the saved model
        """
        modeldir = os.path.join("models", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs("models", exist_ok=True)

        model_path = modeldir
        if suffix:
            model_path += f"-{suffix}"
        model_path += ".h5"

        print(f"Saving model to: {model_path}")
        model.save(model_path)
        return model_path

    @staticmethod
    def load_model(model_path: str) -> tf_keras.Model:
        """
        Load a saved model.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded Keras model
        """
        print(f"Loading saved model from: {model_path}")
        model = tf_keras.models.load_model(
            model_path,
            custom_objects={"KerasLayer": hub.KerasLayer}
        )
        return model


class Predictor:
    """Handles predictions and visualization."""

    def __init__(self, unique_breeds: np.ndarray):
        """
        Initialize the predictor.

        Args:
            unique_breeds: Array of unique breed names
        """
        self.unique_breeds = unique_breeds

    def get_pred_label(self, prediction_probabilities: np.ndarray) -> str:
        """
        Convert prediction probabilities to a label.

        Args:
            prediction_probabilities: Array of prediction probabilities

        Returns:
            Predicted breed label
        """
        return self.unique_breeds[np.argmax(prediction_probabilities)]

    def unbatchify(self, data: tf.data.Dataset) -> Tuple[List[np.ndarray], List[str]]:
        """
        Unbatch a dataset into separate images and labels.

        Args:
            data: Batched dataset

        Returns:
            Tuple of (images, labels)
        """
        images = []
        labels = []

        for image, label in data.unbatch().as_numpy_iterator():
            images.append(image)
            labels.append(self.get_pred_label(label))

        return images, labels

    def plot_prediction(self, prediction_probabilities: np.ndarray,
                       labels: List[str],
                       images: List[np.ndarray],
                       n: int = 0):
        """
        Plot a single prediction with its image and label.

        Args:
            prediction_probabilities: Array of all prediction probabilities
            labels: List of true labels
            images: List of images
            n: Index of the prediction to plot
        """
        pred_prob, true_label, image = prediction_probabilities[n], labels[n], images[n]
        pred_label = self.get_pred_label(pred_prob)

        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])

        color = "green" if pred_label == true_label else "red"
        plt.title(f"{pred_label} {np.max(pred_prob)*100:.0f}% {true_label}", color=color)

    def plot_prediction_confidence(self, prediction_probabilities: np.ndarray,
                                   labels: List[str],
                                   n: int = 0,
                                   top_k: int = 10):
        """
        Plot top K prediction confidences as a bar chart.

        Args:
            prediction_probabilities: Array of all prediction probabilities
            labels: List of true labels
            n: Index of the prediction to plot
            top_k: Number of top predictions to show
        """
        pred_prob, true_label = prediction_probabilities[n], labels[n]
        pred_label = self.get_pred_label(pred_prob)

        # Get top K predictions
        top_k_indexes = pred_prob.argsort()[-top_k:][::-1]
        top_k_values = pred_prob[top_k_indexes]
        top_k_labels = self.unique_breeds[top_k_indexes]

        # Plot
        top_plot = plt.bar(np.arange(len(top_k_labels)), top_k_values, color="grey")
        plt.xticks(np.arange(len(top_k_labels)), labels=top_k_labels, rotation="vertical")

        # Highlight true label if in top K
        if np.isin(true_label, top_k_labels):
            top_plot[np.argmax(top_k_labels == true_label)].set_color("green")

    def show_images_grid(self, images: List[np.ndarray],
                        labels: List[str],
                        grid_size: int = 5):
        """
        Display a grid of images with their labels.

        Args:
            images: List of images
            labels: List of labels
            grid_size: Size of the grid (grid_size x grid_size)
        """
        plt.figure(figsize=(10, 10))

        for i in range(min(grid_size * grid_size, len(images))):
            ax = plt.subplot(grid_size, grid_size, i + 1)
            plt.imshow(images[i])
            plt.title(labels[i] if isinstance(labels[i], str) else self.unique_breeds[labels[i].argmax()])
            plt.axis("off")

    def create_kaggle_submission(self, test_predictions: np.ndarray,
                                test_filenames: List[str],
                                output_path: str = "dog_breed_predictions_kaggle_format.csv"):
        """
        Create a Kaggle submission file from test predictions.

        Args:
            test_predictions: Array of prediction probabilities
            test_filenames: List of test image filenames
            output_path: Path to save the submission file
        """
        # Create DataFrame
        preds_df = pd.DataFrame(columns=["id"] + list(self.unique_breeds))
        preds_df["id"] = [os.path.basename(fname).split(".")[0] for fname in test_filenames]

        # Add prediction probabilities
        for i, breed in enumerate(self.unique_breeds):
            preds_df[breed] = test_predictions[:, i]

        # Save to CSV
        preds_df.to_csv(output_path, index=False)
        print(f"Submission file saved to: {output_path}")


class DogBreedClassifier:
    """Main class that orchestrates the entire dog breed classification pipeline."""

    def __init__(self, labels_csv_path: str = "labels.csv",
                 train_dir: str = "train/",
                 img_size: int = 224,
                 batch_size: int = 32,
                 model_url: str = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/5"):
        """
        Initialize the dog breed classifier.

        Args:
            labels_csv_path: Path to the labels CSV file
            train_dir: Directory containing training images
            img_size: Target image size for resizing
            batch_size: Batch size for data batching
            model_url: TensorFlow Hub model URL
        """
        self.preprocessor = DataPreprocessor(labels_csv_path, train_dir, img_size, batch_size)
        self.model_builder = ModelBuilder(img_size, model_url)
        self.trainer = ModelTrainer()
        self.model_manager = ModelManager()

        self.model = None
        self.unique_breeds = None
        self.predictor = None
        self.train_data = None
        self.val_data = None

    def prepare_data(self, num_images: Optional[int] = None, test_size: float = 0.2):
        """
        Prepare training and validation data.

        Args:
            num_images: Number of images to use (None for all)
            test_size: Proportion of data for validation
        """
        print("Preparing data...")
        result = self.preprocessor.prepare_train_val_split(num_images, test_size)
        self.train_data, self.val_data, X_train, X_val, y_train, y_val = result
        self.unique_breeds = self.preprocessor.unique_breeds
        self.predictor = Predictor(self.unique_breeds)
        print(f"Data prepared: {len(X_train)} training, {len(X_val)} validation samples")
        print(f"Number of unique breeds: {len(self.unique_breeds)}")

    def build_model(self):
        """Build the classification model."""
        if self.unique_breeds is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        print("Building model...")
        self.model = self.model_builder.create_model(len(self.unique_breeds))
        print("Model built successfully")
        return self.model

    def train(self, epochs: int = 10, use_early_stopping: bool = True):
        """
        Train the model.

        Args:
            epochs: Number of training epochs
            use_early_stopping: Whether to use early stopping
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print(f"Training model for {epochs} epochs...")
        self.model = self.trainer.train_model(
            self.model,
            self.train_data,
            self.val_data,
            epochs=epochs,
            use_early_stopping=use_early_stopping
        )
        print("Training complete")

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the model on validation data.

        Returns:
            Tuple of (loss, accuracy)
        """
        if self.model is None or self.val_data is None:
            raise ValueError("Model not trained or validation data not available.")

        print("Evaluating model...")
        loss, accuracy = self.model.evaluate(self.val_data)
        print(f"Validation Loss: {loss:.4f}, Validation Accuracy: {accuracy:.4f}")
        return loss, accuracy

    def predict(self, data: tf.data.Dataset) -> np.ndarray:
        """
        Make predictions on a dataset.

        Args:
            data: Dataset to make predictions on

        Returns:
            Array of prediction probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model.predict(data, verbose=1)

    def save(self, suffix: Optional[str] = None) -> str:
        """
        Save the trained model.

        Args:
            suffix: Optional suffix for the filename

        Returns:
            Path to the saved model
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        return self.model_manager.save_model(self.model, suffix)

    def load(self, model_path: str):
        """
        Load a saved model.

        Args:
            model_path: Path to the saved model
        """
        self.model = self.model_manager.load_model(model_path)

    def prepare_test_data(self, test_dir: str = "test/") -> Tuple[tf.data.Dataset, List[str]]:
        """
        Prepare test data for predictions.

        Args:
            test_dir: Directory containing test images

        Returns:
            Tuple of (test_data, test_filenames)
        """
        test_filenames = [test_dir + fname for fname in os.listdir(test_dir)]
        test_data = self.preprocessor.create_data_batches(test_filenames, test_data=True)
        return test_data, test_filenames

    def create_submission(self, test_dir: str = "test/",
                         output_path: str = "dog_breed_predictions_kaggle_format.csv"):
        """
        Create a Kaggle submission file.

        Args:
            test_dir: Directory containing test images
            output_path: Path to save the submission file
        """
        if self.predictor is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")

        print("Preparing test data...")
        test_data, test_filenames = self.prepare_test_data(test_dir)

        print("Making predictions on test data...")
        test_predictions = self.predict(test_data)

        print("Creating submission file...")
        self.predictor.create_kaggle_submission(test_predictions, test_filenames, output_path)


def main():
    """Example usage of the DogBreedClassifier."""
    # Check GPU availability
    print("GPU Available!" if tf.config.list_physical_devices("GPU") else "GPU NOT Available!")
    print(f"TensorFlow version: {tf.__version__}")

    # Initialize classifier
    classifier = DogBreedClassifier()

    # Prepare data (using subset for faster training)
    classifier.prepare_data(num_images=4000, test_size=0.2)

    # Build and train model
    classifier.build_model()
    classifier.train(epochs=10, use_early_stopping=True)

    # Evaluate
    classifier.evaluate()

    # Save model
    model_path = classifier.save(suffix="mobilenetv2-4000images")
    print(f"Model saved to: {model_path}")

    # Optional: Create Kaggle submission
    # classifier.create_submission()


if __name__ == "__main__":
    main()
