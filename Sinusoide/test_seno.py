import keras
from sin_losses import sin_error_fcn
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
model = keras.models.load_model("sine_model.h5", custom_objects={"sin_error_fcn": sin_error_fcn})

df = pd.read_csv("sine_data.csv")
x_test = df.iloc[67].values.astype(np.float32).reshape(1, -1)  # una fila con shape (1, 100)

t_2 = tf.linspace(0.0, 1.0, 44100)[0:100]  # mismos puntos que en el dataset original

x_test_2 = tf.sin(2 * np.pi * 150* t_2)  # seno de 440 Hz
x_test_2 = tf.reshape(x_test_2, (1, -1))  # shape (1, 100)

pred = model.predict(x_test)
fase = pred[0][1] * 2 * np.pi  # desnormalización de fase
freq = pred[0][0] * 440        # desnormalización de frecuencia

print("Frecuencia (Hz):", freq)
print("Fase (rad):", fase)

def plot_result(y_true, y_pred):
    freq = y_pred[:, 0] *440        # desnormalización de frecuencia
    phase = y_pred[:, 1] * 2 * np.pi  # desnormalización de fase

    t = tf.linspace(0.0, 1.0, 44100)[0:100]  # mismos puntos que en el dataset original
    t = tf.reshape(t, (1, -1))  # shape (1, 100)

    freq = tf.reshape(freq, (-1, 1))   # (batch, 1)
    phase = tf.reshape(phase, (-1, 1)) # (batch, 1)

    seno = tf.sin(2 *np.pi *freq * t + phase)  # reconstrucción: shape (1, 100)

    print("seno: ", seno.shape)  # (1, 100)
    print("y_true: ", y_true.shape)  # (1, 100)
    
    #Estos graficos los use para debuggear no son necesarios al usar la funcion
    plt.plot(t[0], seno[0], label="seno predicho")
    plt.plot(t[0], y_true[0], label="seno real")
    plt.legend()
    plt.grid()
    plt.title("Comparación entre señal real y reconstruida")
    plt.show()

plot_result(x_test_2, pred)
