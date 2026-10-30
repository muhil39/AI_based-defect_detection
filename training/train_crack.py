"""
Training script for Model 2: Surface Crack Detection CNN.
Classifies metal surfaces into: Crack vs. No Crack.
Saves the trained model to 'models/crack_detector.keras'.
"""

import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "cracks")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "crack_detector.keras")

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-3


def verify_dataset(dataset_dir: str):
    """Checks that classes crack and no_crack exist and contain images."""
    required_classes = ["crack", "no_crack"]
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

    print("\nDataset Verification (Surface Cracks):")
    for cls, count in counts.items():
        print(f" - {cls.upper()}: {count} images")

    total_images = sum(counts.values())
    if total_images == 0:
        print("\n[ERROR] No training images found in dataset/cracks/!")
        print("To populate with sample dataset for immediate testing, run:")
        print("    python training/generate_sample_data.py")
        print("Or place your dataset images into:")
        for cls in required_classes:
            print(f"    dataset/cracks/{cls}/")
        sys.exit(1)

    return counts


def build_crack_detector_cnn(input_shape=(224, 224, 3)):
    """
    Builds a high-sensitivity CNN specialized for micro-crack detection.
    Features smaller initial filter sizes to detect sharp edge discontinuities,
    residual connections, and spatial dropout to prevent overfitting on clean metal textures.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.15),
        layers.RandomContrast(0.2),
    ], name="crack_augmentation")

    model = models.Sequential([
        layers.Input(shape=input_shape),
        data_augmentation,

        # Block 1 - Fine feature extraction (cracks are thin lines)
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Block 4
        layers.Conv2D(256, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        # Dense Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        # 2 outputs: [crack_score, no_crack_score]
        layers.Dense(2, activation="softmax", name="crack_output")
    ], name="Surface_Crack_Detector_CNN")

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
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # Load validation set (20%)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=IMG_SIZE,
        batch_size=batch_size,
        label_mode="categorical"
    )

    class_names = train_ds.class_names
    print(f"Detected Class Mapping: {class_names}")

    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Build model
    model = build_crack_detector_cnn(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
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

    print(f"\nBeginning Crack Detection CNN Training ({epochs} epochs)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )

    print(f"\nSaving trained model to: {MODEL_SAVE_PATH}")
    model.save(MODEL_SAVE_PATH)
    print("Crack Detector Model saved successfully!")

    val_loss, val_acc = model.evaluate(val_ds)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.2f}% | Loss: {val_loss:.4f}")

    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Surface Crack Detector CNN")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    train(epochs=args.epochs, batch_size=args.batch_size)

