import keras
import numpy as np
import matplotlib.pyplot as plt

#input: sinusoide cruda
# nosotros: probemos pasarle la sinusoide o pasarle fft o pasarle la stft -> parametros FM

def build_model(input_shape):
    inputs = keras.layers.Input(shape=input_shape)

    # Backbone
    x = keras.layers.Dense(256, activation='relu')(inputs)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(64, activation='relu')(x)

    # Amplitude head
    amp = keras.layers.Dense(32, activation='relu')(x)
    amp = keras.layers.Dense(16, activation='relu')(amp)
    amp_output = keras.layers.Dense(15, activation='sigmoid', name='amplitudes')(amp)

    # Frequency head
    freq = keras.layers.Dense(32, activation='relu')(x)
    freq = keras.layers.Dense(16, activation='relu')(freq)
    freq_output = keras.layers.Dense(15, activation='sigmoid', name='frequencies')(freq)

    model = keras.models.Model(inputs=inputs, outputs=[amp_output, freq_output])
    model.compile(
        optimizer= keras.optimizers.Adam(),
        loss={'amplitudes': keras.losses.MeanSquaredError(), 'frequencies': keras.losses.MeanSquaredError()},
        metrics={'amplitudes': 'mse', 'frequencies': 'mse'}
    )

    return model

def train_model(model, x_train, y_train, epochs=10, batch_size=32):
    history = model.fit(
        x_train, 
        y_train, 
        epochs=epochs, 
        batch_size=batch_size,
        validation_split=0.2)
    
    return history

if __name__ == "__main__":
 
    df = np.load("dataset.npz") #cargar datos

    x_train = df["X"][:, 0:5000] #Espectros de frecuencias
    y_train = df["Y"] #Peaks de frecuencias

    y_train = {
        'amplitudes': y_train[:, :, 0],  
        'frequencies': y_train[:, :, 1]  
    }

    input_shape = (x_train.shape[1],)
    print(input_shape)

    model = build_model(input_shape) 
    model.summary() 

    history = train_model(model, x_train, y_train, epochs=100) #Entrenar el modelo

    # Graficar la pérdida de entrenamiento y validación
    plt.plot(history.history["loss"], label="Pérdida de entrenamiento")
    plt.plot(history.history["val_loss"], label="Pérdida de validación")
    plt.xlabel("Época")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Curvas de pérdida")
    plt.show()

    model.save("peaks_model_multihead.h5") #guardo