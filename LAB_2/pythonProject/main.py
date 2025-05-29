import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 1. Wczytanie i normalizacja danych
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0

# 2. Wizualizacja 20 przykładów ze zbioru treningowego
n_vis = 20
plt.figure(figsize=(8,4))
for i in range(n_vis):
    plt.subplot(2, n_vis//2, i+1)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.suptitle('Przykłady CIFAR-10 (train)')
plt.show()

# 3. Budowa konwolucyjnego autoenkodera
input_img = keras.Input(shape=(32,32,3))
# Encoder
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2,2), padding='same')(x)         # 16x16x32
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2,2), padding='same')(x)         # 8x8x16 = 1024
encoded = layers.Conv2D(8, (3,3), activation='relu', padding='same', name='code_layer')(x)
encoded = layers.MaxPooling2D((2,2), padding='same')(encoded)  # 4x4x8 = 128 (<512)

# Decoder
x = layers.Conv2D(8, (3,3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2,2))(x)                           # 8x8x8
x = layers.Conv2D(16, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)                           # 16x16x16
x = layers.Conv2D(32, (3,3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2,2))(x)                           # 32x32x32
decoded = layers.Conv2D(3, (3,3), activation='sigmoid', padding='same')(x)  # 32x32x3

autoencoder = keras.Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.SGD(), loss='mse')
autoencoder.summary()

# 4. Trenowanie autoenkodera
history = autoencoder.fit(
    x_train, x_train,
    epochs=100,
    batch_size=1024,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# 5. Wykres strat treningowej i walidacyjnej
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Loss podczas treningu autoenkodera')
plt.show()

# 6. Wizualizacja oryginał vs rekonstrukcja
n_disp = 15
decoded_imgs = autoencoder.predict(x_test[:n_disp])
plt.figure(figsize=(12,4))
for i in range(n_disp):
    # oryginał
    plt.subplot(2, n_disp, i+1)
    plt.imshow(x_test[i])
    plt.axis('off')
    # rekonstrukcja
    plt.subplot(2, n_disp, n_disp+i+1)
    plt.imshow(decoded_imgs[i])
    plt.axis('off')
plt.suptitle('Oryginał vs Rekonstrukcja')
plt.show()

# 7. Budowa klasyfikatora z zamrożonym enkoderem
encoder = keras.Model(inputs=input_img, outputs=encoded)
encoder.trainable = False

# klasyfikator
clf = keras.Sequential([
    encoder,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')
])
clf.compile(optimizer=tf.keras.optimizers.SGD(),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=['accuracy'])
clf.summary()

# Przygotowanie etykiet
y_train_cat = keras.utils.to_categorical(y_train, 10)
y_test_cat  = keras.utils.to_categorical(y_test, 10)

# 8. Trenowanie klasyfikatora
clf_history = clf.fit(
    x_train, y_train_cat,
    epochs=50,
    batch_size=1024,
    validation_data=(x_test, y_test_cat)
)

# 9. Ewaluacja klasyfikatora
test_loss, test_acc = clf.evaluate(x_test, y_test_cat, batch_size=1024)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")
print(f"Poprawnie sklasyfikowano: {int(test_acc * x_test.shape[0])} / {x_test.shape[0]}")
