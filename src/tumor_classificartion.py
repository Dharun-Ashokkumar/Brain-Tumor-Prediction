"""
Tumor Classification using CNN in TensorFlow/Keras
----------------------------------------------------
This script trains a Convolutional Neural Network to classify MRI images
into three categories: BENIGN, MALIGNANT, NORMAL.

Author: Dharun Ashokkumar
Dataset: Tumor_Dataset (Google Drive)
"""

# ==============================
# 1. Import Required Libraries
# ==============================
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Activation, Dropout, Dense, Flatten, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from PIL import Image
from skimage import transform
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ==============================
# 2. Mount Google Drive & Set Paths
# ==============================
from google.colab import drive
drive.mount('/content/drive')

base_path = '/content/drive/My Drive/Tumour_Dataset_intern'
train_data_dir = os.path.join(base_path, 'Train')
test_data_dir = os.path.join(base_path, 'Test')

# ==============================
# 3. Image Parameters
# ==============================
img_width, img_height = 64, 64
nb_train_samples = 1050
nb_validation_samples = 144
epochs = 10
batch_size = 32

# Input shape configuration
if tf.keras.backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# ==============================
# 4. Data Augmentation & Normalization
# ==============================
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.3,
    zoom_range=0.3,
    width_shift_range=0.25,
    height_shift_range=0.25,
    horizontal_flip=True,
    vertical_flip=True
)
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_batches = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=['BENIGN', 'MALIGNANT', 'NORMAL']
)
test_batches = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    classes=['BENIGN', 'MALIGNANT', 'NORMAL']
)

# ==============================
# 5. Build CNN Model
# ==============================
model = Sequential([
    Conv2D(32, (3, 3), input_shape=input_shape, padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), padding='same'),
    Activation('relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(64), Activation('relu'),
    Dense(128), Activation('relu'),
    Dropout(0.8),
    Dense(3), Activation('softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ==============================
# 6. Train Model
# ==============================
history = model.fit(
    train_batches,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=test_batches,
    validation_steps=nb_validation_samples // batch_size,
    callbacks=[
        tf.keras.callbacks.ModelCheckpoint(
            filepath='Model_{val_accuracy:.3f}.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
)

# ==============================
# 7. Plot Training & Validation Accuracy
# ==============================
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot Loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# ==============================
# 8. Confusion Matrix
# ==============================
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45)
    plt.yticks(ticks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Evaluate confusion matrix
test_batches_cm = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=nb_validation_samples,
    classes=['BENIGN', 'MALIGNANT', 'NORMAL']
)
test_imgs, test_labels = next(test_batches_cm)
predictions = np.argmax(model.predict(test_batches_cm, steps=1), axis=-1)
true_labels = np.argmax(test_labels, axis=-1)
cm = confusion_matrix(true_labels, predictions)
plot_confusion_matrix(cm, classes=['BENIGN', 'MALIGNANT', 'NORMAL'])

# docs/accuracy.png
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('/content/docs/accuracy.png')  # Save into docs/
plt.close()

# docs/loss.png
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('/content/docs/loss.png')
plt.close()

# docs/confusion_matrix.png
cm = confusion_matrix(true_labels, predictions)

plt.figure()
plot_confusion_matrix(cm, classes=['BENIGN', 'MALIGNANT', 'NORMAL'])
plt.savefig('/content/docs/confusion_matrix.png')
plt.close()

# ==============================
# 9. Save Model
# ==============================
model.save('tumour_classification_model.h5')

# ==============================
# 10. Predict Single Image
# ==============================
def load_image(filename):
    img = Image.open(filename)
    img = img.resize((64, 64))
    img = np.array(img).astype('float32') / 255.0
    if img.ndim == 2:  # grayscale to RGB
        img = np.stack([img]*3, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

# Example prediction
image_path = '/content/drive/My Drive/path/to/your/image.jpg'
image = load_image(image_path)
prediction = model.predict(image)
predicted_class = np.argmax(prediction, axis=-1)[0]

if predicted_class == 0:
    print("Benign")
elif predicted_class == 1:
    print("Malignant")
else:
    print("Normal")
