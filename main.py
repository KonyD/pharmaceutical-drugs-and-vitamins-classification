import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers import Rescaling, Resizing, Normalization, RandomFlip, RandomRotation

from tensorflow.keras.applications import MobileNetV2

from pathlib import Path
import os.path

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings("ignore")

# Path to dataset folder
dataset = "Drug Vision/Data Combined"
image_dir = Path(dataset)

# Collect all image file paths with .jpg and .png extensions
filepaths = list(image_dir.glob(r"**/*.jpg")) + list(image_dir.glob(r"**/*.png"))

# Extract labels from folder names
labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

# Convert file paths and labels into pandas Series
filepaths = pd.Series(filepaths, name="filepath").astype("str")
labels = pd.Series(labels, name="label")

# Combine file paths and labels into a DataFrame
image_df = pd.concat([filepaths, labels], axis=1)

# Display a random sample of 25 images from the dataset
random_indices = np.random.randint(0, len(image_df), size=25)
fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(11, 11))
for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_indices[i]]))
    ax.set_title(image_df.label[random_indices[i]])
plt.tight_layout()
plt.show()

# Split the dataset into training and testing sets
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42, shuffle=True)

# Create ImageDataGenerators for data augmentation and preprocessing
train_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    validation_split=0.2
)

test_generator = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
)

# Load training images and apply preprocessing
train_images = train_generator.flow_from_dataframe(
    dataframe=image_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="training"
)

# Load validation images and apply preprocessing
val_images = train_generator.flow_from_dataframe(
    dataframe=train_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
    shuffle=True,
    seed=42,
    subset="validation"
)

# Load test images
test_images = test_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col="filepath",
    y_col="label",
    target_size=(224, 224),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=64,
)

# Create a preprocessing layer for resizing and rescaling images
resize_and_rescale = tf.keras.Sequential([
    Resizing(224, 224),
    Rescaling(1.0/255)
])

# Load the pretrained MobileNetV2 model without the top classification layer
pretrained_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg"
)

# Freeze the weights of the pretrained model to prevent training
pretrained_model.trainable = False

# Define callbacks for early stopping and model checkpointing
checkpoint_path = "pharmacceutical_drugs_and_vitamins_classification_model_checkpoint.weights.h5"
checkpoint_callback = ModelCheckpoint(checkpoint_path,
                save_weights_only=True,
                monitor="val_accuracy",
                save_best_only=True
                )

early_stopping = EarlyStopping(monitor="val_accuracy",
              patience=5,
              restore_best_weights=True
              )

# Define the model architecture
inputs = pretrained_model.input
x = resize_and_rescale(inputs)
x = pretrained_model(x)  # Pass input through pretrained model
x = Dense(256, activation="relu")(x)  # Fully connected layer
x = Dropout(0.2)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
outputs = Dense(10, activation="softmax")(x)  # Output layer for 10 classes

model = Model(inputs=inputs, outputs=outputs)

# Compile the model
model.compile(optimizer=Adam(0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(
    train_images,
    steps_per_epoch=len(train_images),
    validation_data=val_images,
    validation_steps=len(val_images),
    epochs=10,
    callbacks=[early_stopping, checkpoint_callback]
)

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(test_images, verbose=1)
print(f"loss: {loss:.4f}, accuracy: {accuracy:.4f}")

# Plot training and validation accuracy/loss
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], marker="o", label="Training Accuracy")
plt.plot(history.history["val_accuracy"], marker="o", label="Validation Accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], marker="o", label="Training Loss")
plt.plot(history.history["val_loss"], marker="o", label="Validation Loss")
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Make predictions on the test set
pred = model.predict(test_images)
pred = np.argmax(pred, axis=1)  # Convert probabilities to class indices

# Map class indices back to their labels
labels = (train_images.class_indices)
labels = dict((v, k) for k, v in labels.items())
pred = [labels[k] for k in pred]

# Visualize predictions vs true labels
random_index = np.random.randint(0, len(test_df) - 1, 15)
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(11, 11)) 

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(image_df.filepath[random_index[i]]))
    if test_df.label.iloc[random_index[i]] == pred[random_index[i]]:
        color = "green"
    else:
        color = "red"
    
    ax.set_title(f"True: {test_df.label.iloc[random_index[i]]} \n Predicted: {pred[random_index[i]]}", color=color)

plt.tight_layout()
plt.show()

# Print a classification report for the test set
y_test = list(test_df.label)
print(classification_report(y_test, pred))
