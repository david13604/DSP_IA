import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sin_error_fcn(y_true, y_pred):
    freq = y_pred[:, 0] *440        # desnormalización de frecuencia
    phase = y_pred[:, 1] * 2 * np.pi  # desnormalización de fase

    t = tf.linspace(0.0, 1.0, 44100)[0:100]  # mismos puntos que en el dataset original
    t = tf.reshape(t, (1, -1))  # shape (1, 100)

    freq = tf.reshape(freq, (-1, 1))   # (batch, 1)
    phase = tf.reshape(phase, (-1, 1)) # (batch, 1)

    seno = tf.sin(2 *np.pi *freq * t + phase)  # reconstrucción: shape (1, 100)

    print("seno: ", seno.shape)  # (1, 100)
    print("y_true: ", y_true.shape)  # (1, 100)
    """
    #Estos graficos los use para debuggear no son necesarios al usar la funcion
    plt.plot(t[0], seno[0], label="seno predicho")
    plt.plot(t[0], y_true[0], label="seno real")
    plt.legend()
    plt.grid()
    plt.title("Comparación entre señal real y reconstruida")
    plt.show()
    """
    return tf.reduce_mean(tf.square(y_true - seno), axis=-1)

if __name__ == "__main__":
    df = pd.read_csv("sine_data.csv")

    # 20 muestras de la sinusoide con fase 30 y frecuencia 5 Hz
    y = df.iloc[0].values  # shape (100,)
    y = tf.convert_to_tensor(y, dtype=tf.float32)
    y = tf.reshape(y, (1, -1))  # reshape para que tenga shape (1, 100)
    
    # Predicción ideal: frecuencia 5Hz, fase 30°
    frec = 440/440     # normalizado
    phase = 30 / 360  # normalizado
    y_pred = tf.convert_to_tensor([[frec, phase]], dtype=tf.float32)
    error = sin_error_fcn(y, y_pred)
    print("Error: ", error.numpy())  
