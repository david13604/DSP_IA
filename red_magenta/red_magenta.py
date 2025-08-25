import keras
import os
import tensorflow as tf

"""
Queda por hacer:
- Darle el audio de input es decir input_shape desaparece
  Este ultimo sale de una funcion (por crear) que calcule MFCCs
- Crear otra funcion que calcule F0 y Loudness
- Actualmente f_in y l_in no salen de ningun lado por eso el punto anterior
- A partir del audio sacar stft de este como del sintetizado y calcular loss entre ambos
- Probar todo :D
"""
os.environ["KERAS_BACKEND"] = "tensorflow"

GRU_UNITS = 512

class Autoencder:
    def __init__(self,
                 input_shape=(None, 30),
                 z_dim=16):
        
        self.input_shape = input_shape
        self.z_dim = z_dim

        self.autoencoder = None

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build(self):
        mfcc_in = keras.Input(shape=self.input_shape, name="mfcc_in")
        f_in = keras.Input(shape=(None, 1), name="f_in")        # F0
        l_in = keras.Input(shape=(None, 1), name="l_in")        # Loudness

        z = self.encoder(mfcc_in)
        harmonics, noise = self.decoder([f_in, l_in, z])

        self.autoencoder = keras.Model([mfcc_in, f_in, l_in], [harmonics, noise], name="autoencoder")
        self.autoencoder.summary()
        return self.autoencoder
    
    def mpl_block(self, units=512, layers_num=3, name_prefix="mlp"):
        def block(x):
            for i in range(layers_num):
                x = keras.layers.LayerNormalization(name=f"{name_prefix}_ln{i}")(x)
                x = keras.layers.ReLU(name=f"{name_prefix}_relu{i}")(x)
                x = keras.layers.Dense(units, name=f"{name_prefix}_dense{i}")(x)
            return x
        return block
    
    def build_encoder(self, GRU_UNITS=512):
        # Entrada: (batch, time, features) = (B, T, 30)
        mfcc_in = keras.Input(shape=self.input_shape, name="mfcc_in")

        # GRU a tiempo completo → salida: (B, T, 512)
        x = keras.layers.GRU(GRU_UNITS, return_sequences=True, name="z_gru")(mfcc_in)

        # Proyección por frame a z(t) → salida: (B, T, 16)
        z_out = keras.layers.Dense(self.z_dim, name="z_dense")(x)

        encoder = keras.Model(mfcc_in, z_out, name="z_encoder")
        encoder.summary()

        return encoder
    
    def build_decoder(self):
        # Entradas
        f_in = keras.Input(shape=(None, 1), name="f_in")        # F0
        l_in = keras.Input(shape=(None, 1), name="l_in")        # Loudness
        z_in = keras.Input(shape=(None, self.z_dim), name="z_in")       # Z embedding

        # MLPs individuales
        f_mlp = self.mpl_block(name_prefix="f")(f_in)   # (B, T, 512)
        l_mlp = self.mpl_block(name_prefix="l")(l_in)
        z_mlp = self.mpl_block(name_prefix="z")(z_in)

        # Concat inicial → (B, T, 1536)
        concat1 = keras.layers.Concatenate(name="concat1")([f_mlp, l_mlp, z_mlp])
        
        # GRU → (B, T, 512)
        gru_out = keras.layers.GRU(512, return_sequences=True, name="decoder_gru")(concat1)

        # Concat con f_mlp y l_mlp → (B, T, 1536)
        concat2 = keras.layers.Concatenate(name="concat2")([gru_out, f_mlp, l_mlp])

        # MLP final
        final_mlp = self.mpl_block(name_prefix="final")(concat2)  # (B, T, 512)

        # Dos salidas densas
        harmonics = keras.layers.Dense(101, activation="softplus", name="harmonics")(final_mlp)
        noise = keras.layers.Dense(65, activation="softplus", name="noise")(final_mlp)

        # Modelo
        decoder = keras.Model([f_in, l_in, z_in], [harmonics, noise], name="decoder")
        decoder.summary()
        return decoder

