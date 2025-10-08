import keras
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
from tensorflow import keras

class Autoencder:
    def __init__(self,
                 input_shape=(None, 30),
                 z_dim=16,
                 n_harmonics=101,
                 sample_rate=16000,
                 n_samples=64000,
                 stft_frame_length=1024,
                 stft_frame_step=256):
        
        self.input_shape = input_shape     # (T, 30)
        self.z_dim = z_dim
        self.n_harmonics = n_harmonics
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.stft_frame_length = stft_frame_length
        self.stft_frame_step = stft_frame_step

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.autoencoder = None

    # ------------------------
    # Bloque MLP 3x512
    # ------------------------
    def mpl_block(self, units=512, layers_num=3, name_prefix="mlp"):
        def block(x):
            for i in range(layers_num):
                x = keras.layers.LayerNormalization(name=f"{name_prefix}_ln{i}")(x)
                x = keras.layers.ReLU(name=f"{name_prefix}_relu{i}")(x)
                x = keras.layers.Dense(units, name=f"{name_prefix}_dense{i}")(x)
            return x
        return block

    # ------------------------
    # Encoder Z
    # ------------------------
    def build_encoder(self, GRU_UNITS=512):
        mfcc_in = keras.Input(shape=self.input_shape, name="mfcc_in")
        x = keras.layers.GRU(GRU_UNITS, return_sequences=True, name="z_gru")(mfcc_in)
        z_out = keras.layers.Dense(self.z_dim, name="z_dense")(x)
        return keras.Model(mfcc_in, z_out, name="z_encoder")

    # ------------------------
    # Decoder (harmonics + noise)
    # ------------------------
    def build_decoder(self):
        f_in = keras.Input(shape=(None, 1), name="f_in")
        l_in = keras.Input(shape=(None, 1), name="l_in")
        z_in = keras.Input(shape=(None, self.z_dim), name="z_in")

        f_mlp = self.mpl_block(name_prefix="f")(f_in)         # (B,T,512)
        l_mlp = self.mpl_block(name_prefix="l")(l_in)         # (B,T,512)
        z_mlp = self.mpl_block(name_prefix="z")(z_in)         # (B,T,512)

        concat1 = keras.layers.Concatenate(name="concat1")([f_mlp, l_mlp, z_mlp])  # (B,T,1536)
        gru_out = keras.layers.GRU(512, return_sequences=True, name="decoder_gru")(concat1)  # (B,T,512)
        concat2 = keras.layers.Concatenate(name="concat2")([gru_out, f_mlp, l_mlp])         # (B,T,1536)

        final_mlp = self.mpl_block(name_prefix="final")(concat2)  # (B,T,512)
        harmonics = keras.layers.Dense(self.n_harmonics, activation="softplus", name="harmonics")(final_mlp)  # (B,T,K)
        noise = keras.layers.Dense(65, activation="softplus", name="noise")(final_mlp)  # (B,T,65) (no usado en esta loss)

        return keras.Model([f_in, l_in, z_in], [harmonics, noise], name="decoder")

    # ------------------------
    # Construcción del Autoencoder
    # ------------------------
    def build(self):
        mfcc_in = keras.Input(shape=self.input_shape, name="mfcc_in")   # (B,T,30)
        f_in = keras.Input(shape=(None, 1), name="f_in")                # (B,T,1)
        l_in = keras.Input(shape=(None, 1), name="l_in")                # (B,T,1)

        z = self.encoder(mfcc_in)                           # (B,T,16)
        harmonics, noise = self.decoder([f_in, l_in, z])    # (B,T,101), (B,T,65)

        # Empaquetamos lo necesario para la loss: [harmonics,noise, f_in, l_in] → (B,T,101+1+1) = (B,T,103)
        pack = keras.layers.Concatenate(axis=-1, name="pack_for_loss")([harmonics, noise, f_in, l_in])

        # Este modelo tiene UNA salida "pack" que la loss sabe desempaquetar
        self.autoencoder = keras.Model([mfcc_in, f_in, l_in], pack, name="autoencoder")
        self.autoencoder.summary()
        return self.autoencoder

    # ------------------------
    # Síntesis aditiva (vectorizada, con fase acumulada)
    # ------------------------
    def additive_synth(self, f0, loudness, harmonics, noise):
        """
        f0:     (B, T, 1)
        loud:   (B, T, 1)
        harm:   (B, T, K)
        Devuelve: audio (B, N)
        """
        B = tf.shape(f0)[0]
        T = tf.shape(f0)[1]
        K = tf.shape(harmonics)[2]
        N = self.n_samples

        # Upsampling a tasa de audio (ceil y recorte a N para evitar supuestos)
        reps = tf.cast(tf.math.ceil(tf.cast(N, tf.float32) / tf.cast(T, tf.float32)), tf.int32)
        f0_up   = tf.repeat(f0,   repeats=reps, axis=1)[:, :N, :]        # (B,N,1)
        loud_up = tf.repeat(loudness, repeats=reps, axis=1)[:, :N, :]    # (B,N,1)
        harm_up = tf.repeat(harmonics, repeats=reps, axis=1)[:, :N, :]   # (B,N,K)
        noise_up = tf.repeat(noise, repeats=reps, axis=1)[:, :N, :]     # (B,N,65)

        # Fase acumulada: phi[n] = sum_{m<=n} 2*pi*f0[m]/fs
        omega = 2.0 * tf.constant(np.pi, tf.float32) * f0_up[..., 0] / float(self.sample_rate)  # (B,N)
        phi = tf.cumsum(omega, axis=1)                                   # (B,N)
        phi = phi[:, :, tf.newaxis]                                      # (B,N,1)

        # Índices de armónicos k = 1..K
        ks = tf.reshape(tf.range(1, tf.cast(K, tf.int32) + 1, dtype=tf.float32), (1, 1, -1))  # (1,1,K)

        # Señal por armónico y suma final
        phase = phi * ks                             # (B,N,K)
        sig = loud_up * harm_up * tf.sin(phase)     # (B,N,K)
        audio_harm = tf.reduce_sum(sig, axis=-1)         # (B,N)

        # Ruido
        # --- Parámetros básicos ---
        frame_length = 512     # igual o menor que 1024 (STFT)
        frame_step   = 256     # hop de 256 muestras
        n_fft_bins   = 65      # igual que la salida del decoder
        n_frames_est = tf.cast(tf.math.ceil(tf.cast(N, tf.float32) / frame_step), tf.int32)

        # --- Ruido blanco (B, N) ---
        white = tf.random.uniform(tf.shape(audio_harm), -1.0, 1.0)

        # --- STFT del ruido (B, F, T) ---
        noise_stft = tf.signal.stft(
            white,
            frame_length=frame_length,
            frame_step=frame_step,
            fft_length=frame_length,
            window_fn=tf.signal.hann_window,
            pad_end=True
        )  # (B, F, T)

        mag = tf.abs(noise_stft)
        phase = tf.math.angle(noise_stft)
        F = tf.shape(mag)[1]
        T_frames = tf.shape(mag)[2]

        # --- Filtro del decoder ---
        # Ajustamos el tamaño temporal (interpolación si hace falta)
        noise_t = noise  # (B, T_feat, 65)
        noise_t = tf.image.resize(noise_t[..., tf.newaxis], [T_frames, n_fft_bins], method='bilinear')[..., 0]

        # Normalizar el filtro y expandir a los bins reales de la FFT
        filt = noise_t / (1e-6 + tf.reduce_max(noise_t, axis=-1, keepdims=True))  # (B, T_frames, 65)
        filt = tf.transpose(filt, perm=[0, 2, 1])  # (B, 65, T_frames)
        filt = tf.image.resize(filt[..., tf.newaxis], [F, T_frames], method='bilinear')[..., 0]  # (B, F, T)

        # --- Aplicar magnitud del filtro ---
        mag_filtered = mag * filt

        # --- Reconstrucción del espectro complejo ---
        noise_stft_filt = tf.complex(mag_filtered * tf.cos(phase), mag_filtered * tf.sin(phase))

        # --- Inversa de STFT (reconstrucción temporal) ---
        noise_filtered = tf.signal.inverse_stft(
            noise_stft_filt,
            frame_length=frame_length,
            frame_step=frame_step,
            window_fn=tf.signal.hann_window
        )  # (B, N')

        # Recortar/padear a longitud exacta
        Nf = tf.shape(noise_filtered)[1]
        noise_filtered = tf.cond(
            Nf < N,
            lambda: tf.pad(noise_filtered, [[0,0],[0, N - Nf]]),
            lambda: noise_filtered[:, :N]
        )

        # --- Normalizar y combinar ---
        noise_filtered = noise_filtered / (1e-6 + tf.reduce_max(tf.abs(noise_filtered), axis=1, keepdims=True))
        audio = audio_harm + 0.05 * noise_filtered
        audio = audio / (1e-6 + tf.reduce_max(tf.abs(audio), axis=1, keepdims=True))

        return audio

    # ------------------------
    # Loss espectral con síntesis interna
    # y_true: magnitud de STFT real (B, F, Tstft)
    # y_pred: pack = concat([harmonics(101), f0(1), loud(1)])  → (B, Tfeat, 103)
    # ------------------------
    def spectral_loss(self, y_true, y_pred):
        # Desempaquetar
        K = self.n_harmonics
        harmonics = y_pred[:, :, :K]
        noise     = y_pred[:, :, K:K+65]
        f0        = y_pred[:, :, K+65:K+66]
        loud      = y_pred[:, :, K+66:K+67]

        # Síntesis → audio
        audio_pred = self.additive_synth(f0, loud, harmonics, noise)  # (B,N)

        # STFT predicha (magnitud)
        S_pred = tf.abs(tf.signal.stft(
            audio_pred,
            frame_length=self.stft_frame_length,
            frame_step=self.stft_frame_step,
            window_fn=tf.signal.hann_window
        ))  # (B, Fp, Tp)

        # Asegurar shapes compatibles recortando a la intersección
        Fp = tf.shape(S_pred)[1];  Tp = tf.shape(S_pred)[2]
        Ft = tf.shape(y_true)[1];  Tt = tf.shape(y_true)[2]
        Fm = tf.minimum(Fp, Ft)
        Tm = tf.minimum(Tp, Tt)
        S_pred_c = S_pred[:, :Fm, :Tm]
        S_true_c = y_true[:, :Fm, :Tm]   # y_true debería ser magnitud ya

        # L1 en magnitud espectral
        return tf.reduce_mean(tf.abs(S_true_c - S_pred_c))

    # ------------------------
    # Compilar
    # ------------------------
    def compile(self, lr=1e-3):
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(lr),
            loss=self.spectral_loss
        )

    def fit(self, x_train, y_train, batch_size=16, epochs=10, validation_data=None):
        history = self.autoencoder.fit(
            x=x_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data
        )
        return history
    
if __name__ == "__main__":
    # -------------------------
    # Parámetros del test
    # -------------------------
    B = 2           # batch size
    T = 250         # frames (ej: ~1.6s a hop=256 con fs=16k)
    N = 64000       # samples de audio
    F = 513         # bins espectrales (frame_length=1024 → 513 bins)
    Tstft = 251     # frames STFT para ~64000 samples con hop=256
    
    # -------------------------
    # Datos aleatorios de prueba
    # -------------------------
    mfcc_batch = np.random.randn(B,T, 30).astype("float32")
    f0_batch   = np.abs(np.random.randn(B,T, 1).astype("float32")) * 440  # valores ~frecuencia
    loud_batch = np.random.rand(B,T, 1).astype("float32")
    S_true     = np.abs(np.random.randn(B,F, Tstft).astype("float32"))    # espectrograma mag real
    
    # -------------------------
    # Instanciar y compilar
    # -------------------------
    model = Autoencder(
        input_shape=(None, 30),
        z_dim=16,
        n_harmonics=101,
        sample_rate=16000,
        n_samples=N,
        stft_frame_length=1024,
        stft_frame_step=256
    )
    auto = model.build()
    model.compile(lr=1e-3)
    
    # -------------------------
    # Entrenar (test rápido)
    # -------------------------
    history = model.fit(
        x_train=[mfcc_batch, f0_batch, loud_batch],
        y_train=S_true,
        batch_size=B,
        epochs=2
    )
    
    # -------------------------
    # Plot de la pérdida
    # -------------------------
    plt.plot(history.history["loss"], label="train loss")
    plt.legend()
    plt.show()
