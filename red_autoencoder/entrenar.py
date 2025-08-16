import keras
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

os.environ["KERAS_BACKEND"] = "tensorflow"

class Conv_Autoencoder:
    def __init__(self,
                 input_dim, 
                 encoding_dim):
        
        self.input_dim = input_dim #es una stft (65, 65, 1)
        self.encoding_dim = encoding_dim #32 por poner algo despues lo cambio
        self.shape_before_bottleneck = None
        
        self.autoencoder = None #lo creo en build()

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build(self):
        input_stft = keras.Input(shape=self.input_dim)
        encoded = self.encoder(input_stft)
        decoded = self.decoder(encoded)

        self.autoencoder = keras.Model(input_stft, decoded)
        self.autoencoder.compile(optimizer='adam', loss='mse')

    def build_encoder(self):
        input_img = keras.Input(shape=self.input_dim)
        x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = keras.layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)

        self.shape_before_bottleneck = x.shape[1:]

        x = keras.layers.Flatten()(x)
        encoded = keras.layers.Dense(self.encoding_dim, activation='relu')(x)

        encoder = keras.Model(input_img, encoded)
        return encoder

    def build_decoder(self):
        latent_input = keras.Input(shape=(self.encoding_dim,))
        x = keras.layers.Dense(np.prod(self.shape_before_bottleneck), activation='relu')(latent_input)
        x = keras.layers.Reshape(self.shape_before_bottleneck)(x)
        x = keras.layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
        x = keras.layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same')(x)
        x = keras.layers.Cropping2D(((0, 0), (0, 1)))(x)
        decoded = keras.layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')(x)

        decoder = keras.Model(latent_input, decoded)
        return decoder
    
    def fit(self, x_train, epochs=50, batch_size=64, validation_data=None):
        # validation data split
        x_train = shuffle(x_train)
        x_val = x_train[int(0.8 * len(x_train)):]
        x_train = x_train[:int(0.8 * len(x_train))]

        history = self.autoencoder.fit(
            x_train, 
            x_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_data=(x_val, x_val)
        )

        return history

    def summary(self):
        self.encoder.summary()
        self.decoder.summary()
        self.autoencoder.summary()

    def save(self, path):
        self.autoencoder.save(path + '_autoencoder.keras')
        self.encoder.save(path + '_encoder.keras')
        self.decoder.save(path + '_decoder.keras')


if __name__ == "__main__":
    input_dim = (410, 65, 1)
    encoding_dim = 32

    autoencoder = Conv_Autoencoder(input_dim, encoding_dim)
    autoencoder.build()
    autoencoder.summary()

    path = "dataset_single.npz"

    x_train = np.load(path)["X"]
    # Reshape to match input shape
    x_train = np.expand_dims(x_train, axis=-1).astype(
        np.float32
    )  # Add channel dimension
    history = autoencoder.fit(x_train, epochs=20, batch_size=5)

    autoencoder.save("primero")

    # Plot
    plt.plot(history.history["loss"], label="Training Loss")   
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.show()