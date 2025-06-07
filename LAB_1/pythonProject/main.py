# Projekt wykonany w zespole z Maciejem Szubiczukiem

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras


(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
# Podzielenie pikseli żeby były w zakresie [0,1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Przekształcenie na wektor 3072
input_shape = x_train.shape[1:]
flat_dim = np.prod(input_shape)
x_train_flat = x_train.reshape(-1, flat_dim)
x_test_flat  = x_test.reshape(-1, flat_dim)

# Wizualizacja obrazkow
n_examples = 25
plt.figure(figsize=(5,5))
for i in range(n_examples):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(x_train[i])
    plt.axis('off')
plt.suptitle('25 samples from CIFAR-10 training set')
plt.show()

# budowa autoenkodera(warstwy Dense)
input_img = keras.layers.Input(shape=(flat_dim,))
# Encoder
encoded = keras.layers.Dense(2048, activation='relu')(input_img)
encoded = keras.layers.Dense(1024, activation='relu')(encoded)
encoded = keras.layers.Dense(512, activation='relu', name='code_layer')(encoded)
# Decoder
decoded = keras.layers.Dense(1024, activation='relu')(encoded)
decoded = keras.layers.Dense(2048, activation='relu')(decoded)
decoded = keras.layers.Dense(flat_dim, activation='sigmoid')(decoded)
# Model autoenkodera
autoencoder = keras.Model(inputs=input_img, outputs=decoded)
autoencoder.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.MeanSquaredError())
autoencoder.summary()


# Trening autoenkodera
epochs = 10
batch_size = 1024
history = autoencoder.fit(
    x_train_flat, x_train_flat,
    epochs=epochs,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(x_test_flat, x_test_flat)
)

# 6. Wykres strat treningowej i walidacyjnej
plt.figure()
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Autoencoder Training vs Validation Loss')
plt.show()


# 7. Rekonstrukcja i wizualizacja 10 przykładów
n_display = 10
decoded_imgs = autoencoder.predict(x_test_flat[:n_display])

plt.figure(figsize=(20,4))
for i in range(n_display):
    ax = plt.subplot(2, n_display, i + 1)
    plt.imshow(x_test[i])
    plt.title('orig')
    plt.axis('off')
    ax = plt.subplot(2, n_display, i + 1 + n_display)
    rec_img = decoded_imgs[i].reshape(input_shape)
    plt.imshow(rec_img)
    plt.title('recon')
    plt.axis('off')
plt.suptitle('Original vs Reconstructed')
plt.show()

# 8. Budowa klasyfikatora z zamrożonym enkoderem
encoder_model = keras.Model(inputs=input_img, outputs=encoded)
encoder_model.trainable = False

# One-hot encoding etykiet
y_train_cat = keras.utils.to_categorical(y_train, num_classes=10)
y_test_cat  = keras.utils.to_categorical(y_test,  num_classes=10)


# Dodanie dwoch dense, 256 neuronow ReLU i 10-wyjsciowa softmax do klasyfikacji
clf = keras.Sequential([
    encoder_model,
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])
clf.compile(optimizer=tf.keras.optimizers.SGD(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['accuracy'])
clf.summary()

# Trening i ewaluacja klasyfikatora
epochs_clf = 50
batch_size_clf = 1024
clf_history = clf.fit(
    x_train_flat, y_train_cat,
    epochs=epochs_clf,
    batch_size=batch_size_clf,
    validation_data=(x_test_flat, y_test_cat)
)

eval_results = clf.evaluate(x_test_flat, y_test_cat, batch_size=batch_size_clf)
print(f"Test loss: {eval_results[0]:.4f}, Test accuracy: {eval_results[1]:.4f}")
print(f"Correctly classified images: {int(eval_results[1] * x_test_flat.shape[0])} / {x_test_flat.shape[0]}")
