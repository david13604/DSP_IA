import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

fs = 44100  # Frecuencia de muestreo
t = np.arange(44100) / fs  # 1000 muestras, espaciadas en el tiempo

y_30 = np.sin(2*np.pi*440*t + np.pi/6)   # 440 Hz, fase 30°
y_60 = np.sin(2*np.pi*220*t + np.pi/3)   # 220 Hz, fase 60°
y_00 = np.sin(2*np.pi*100*t)            # 100 Hz, sin fase

#aca voy a separar de 100 en 100 muestras

all_sines = {
    "y_30": y_30,
    "y_60": y_60,
    "y_00": y_00
}

muestras = 100
segmentos = []
labels = []

for i in range(0, len(y_30), muestras):
    for label, y in all_sines.items():
        segment = y[i:i+muestras]
        if len(segment) == muestras:
            segmentos.append(segment)
            labels.append(label)

df_segments = pd.DataFrame(segmentos)

df_segments.to_csv("sine_data.csv", index=False) 
# Verificación
if __name__ == "__main__":
    df = pd.read_csv("sine_data.csv")

    plt.plot(df.iloc[0].values)
    plt.grid()
    plt.show()