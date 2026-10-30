"""
Training script for Model 1: Material Identification CNN.
Classifies components into: Metal, Plastic, Other.
Uses TensorFlow/Keras with data augmentation, lightweight CNN/Transfer Learning,
and saves the trained model to 'models/material_classifier.keras'.
"""

import os
import sys
import argparse
import numpy as np

# Set project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "material")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "material_classifier.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3


def verify_dataset(dataset_dir: str):
    """Checks that classes metal, plastic, other exist and contain images."""
    required_classes = ["metal", "plastic", "other"]
    counts = {}
    valid_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    for cls in required_classes:
        cls_dir = os.path.join(dataset_dir, cls)
        if not os.path.exists(cls_dir):
            os.makedirs(cls_dir, exist_ok=True)
            counts[cls] = 0
            continue
        files = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in valid_exts]
        counts[cls] = len(files)

    print("\nDataset Verification (Material):")
    for cls, count in counts.items():
        print(f" - {cls.upper()}: {count} images")

    total_images = sum(counts.values())
    if total_images == 0:
        print("\n[ERROR] No training images found in dataset/material/!")
        print("To populate with sample dataset for immediate testing, run:")
        print("    python training/generate_sample_data.py")
        print("Or place your dataset images into:")
        for cls in required_classes:
            print(f"    dataset/material/{cls}/")
        sys.exit(1)

    return counts


def build_material_cnn(input_shape=(224, 224, 3), num_classes=3):
    """
    Builds a lightweight yet high-accuracy CNN architecture for material surface classification.
    Incorporates data augmentation, batch normalization, depthwise separable convolutions,
    and dropout for generalization.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.15),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ], name="material_augmentation")

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

        # Conv Block 1
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Conv Block 4
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Classifier Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax", name="material_output")
    ], name="Material_Classifier_CNN")

    return model


def train(dataset_dir=DATASET_DIR, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Loads dataset, splits into train/val, trains CNN, and saves model."""
    import tensorflow as tf

    verify_dataset(dataset_dir)
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\nLoading dataset using tf.keras.utils.image_dataset_from_directory...")
    # Load training set (80%)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Load validation set (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"Detected Class Mapping: {class_names}")

    # Normalization layer: [0, 255] -> [0.0, 1.0]
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    # Performance optimization
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    model = build_material_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=len(class_names))
    model.summary()

    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)
    ]

    print(f"\nBeginning CNN Training ({epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    # Save model in modern Keras v3 format (.keras)
    print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Material Classification Model saved successfully!")

    # Final Validation evaluation
    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Material Classifier CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)

