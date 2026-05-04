
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
from tensorflow.keras.datasets import cifar10, mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


try:
    print("Loading CIFAR-10 dataset...")
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    input_shape = (32, 32, 3)
    num_classes = 10

except Exception as e:
    print(" CIFAR-10 gagal, pakai MNIST sebagai fallback...")
    print("Error:", e)

    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # reshape ke format CNN
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    input_shape = (28, 28, 1)
    num_classes = 10

# PREPROCESSING
x_train = x_train / 255.0
x_test = x_test / 255.0

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

print("Train shape:", x_train.shape)
print("Test shape:", x_test.shape)

# BUILD MODEL
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu'),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

# COMPILE
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# TRAINING
history = model.fit(
    x_train, y_train,
    epochs=5,   # biar cepat dulu
    batch_size=32,
    validation_data=(x_test, y_test)
)

# EVALUASI
loss, acc = model.evaluate(x_test, y_test)
print("Test Accuracy:", acc)

# SAVE MODEL
model.save("cnn_cifar10.keras")
print(" Model berhasil disimpan: cnn_cifar10.keras")