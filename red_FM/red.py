import keras 
import os
from scipy.signal import stft
import numpy as np
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

class FM_red:
    def __init__(
            self,
            input_shape,
            output_shape,
    )->None:
        self.input_shape = input_shape
        self.output_shape = output_shape

        self.model = None

        self.build_model()

    def build_model(self):
        input_layer = keras.layers.Input(shape=self.input_shape)

        #Ahora la CNN tipica 
        x = keras.layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(self.output_shape, activation='relu')(x)

        #Ahora divido es sub capas de amplitud, frecuencia carrier e indice de modulacion
        f_c = keras.layers.Dense(self.output_shape//3, activation='sigmoid', name='f_C')(x) #frecuencia carrier
        I = keras.layers.Dense(self.output_shape//3, activation='sigmoid', name='f_M')(x) #indice de modulacion
        A = keras.layers.Dense(self.output_shape//3, activation='sigmoid', name='f_I')(x) #amplitud

        #ahora junto todo porque me pide unsa salida
        output_layer = self.crear_stft(f_c, I, A)
        self.model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    def compile(self, learning_rate=0.0001):
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
        pass 

    def fit(self, x_train, y_train, epochs=10, batch_size=32, validation_data=None):
        self.model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            shuffle=True
        )

    def crear_stft(f_c, I, A, dur=1.0, fs=16000, nperseg=128, noverlap=64):
        # f_c, I, A: (batch,)
        batch_size = tf.shape(f_c)[0]
        n_samples = tf.cast(tf.math.round(dur * fs), tf.int32)
    
        # Tiempo t: (n_samples,)
        t = tf.linspace(0.0, dur, n_samples)  # tf.float32
    
        # Expand to (batch, n_samples)
        t = tf.reshape(t, (1, -1))
        t = tf.tile(t, [batch_size, 1])  # shape (batch, n_samples)
    
        # Expand parameters to match t
        f_c_exp = tf.reshape(f_c, (-1, 1))  # (batch, 1)
        I_exp = tf.reshape(I, (-1, 1))
        A_exp = tf.reshape(A, (-1, 1))
    
        # Señal FM sintetizada: x(t) = A * sin(2πf_c t + I * sin(2πf_c t))
        phi = 2.0 * tf.constant(tf.math.pi) * f_c_exp * t
        mod = I_exp * tf.sin(phi)
        x = A_exp * tf.sin(phi + mod)  # (batch, n_samples)
    
        # STFT: usa tf.signal.stft
        stft_result = tf.signal.stft(
            x,
            frame_length=nperseg,
            frame_step=nperseg - noverlap,
            fft_length=nperseg,
            window_fn=tf.signal.hann_window
        )  # shape: (batch, frames, freq_bins)
    
        # Magnitud en dB
        magnitude = tf.abs(stft_result)
        magnitude_db = 20.0 * tf.math.log(magnitude + 1e-6) / tf.math.log(10.0)
    
        # Normalización por batch
        min_val = tf.reduce_min(magnitude_db, axis=[1, 2], keepdims=True)
        max_val = tf.reduce_max(magnitude_db, axis=[1, 2], keepdims=True)
        magnitude_db_norm = (magnitude_db - min_val) / (max_val - min_val + 1e-6)
    
        return magnitude_db_norm  # shape: (batch, frames, freq_bins)
    
    def crear_stft(self, f_C, I, A, fs = 44100, cant_muestras = 4096):
        t = np.arange(cant_muestras) / fs
        
        for f_c, i, a in zip(f_C, I, A):
            fm_signal = a * np.sin(2 * np.pi * f_c * t + i * np.sin(2 * np.pi * f_c * t))

        f, t, Zxx = stft(fm_signal, fs=fs, nperseg=1024)
        Zxx_mag = np.abs(Zxx)
        Zxx_mag = 20 * np.log10(Zxx_mag + 1e-6)
        min_val = np.min(Zxx_mag)
        max_val = np.max(Zxx_mag)
        Zxx_mag = (Zxx_mag - min_val) / (max_val - min_val)

        return Zxx_mag
    
if __name__ == "__main__":
    input_shape = (65, 65, 1)  
    output_shape = 18  
    
    model = FM_red(input_shape, output_shape)
    model.compile()
    
    # Example data for training
    x_train = np.random.rand(100, *input_shape)  # 100 samples of random data
    
    model.fit(x_train, x_train, epochs=5, batch_size=10)