import torch
import librosa
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import time
torch.set_grad_enabled(False)

T_LATENTE = 1000
LATENT_DIM = 8

def graficar_latente(z)-> None:
    z_np = z.squeeze(0).cpu().numpy()  # shape: (8, T_latente)

    latent_dim, T_latent = z_np.shape   

    fig, axes = plt.subplots(
        latent_dim, 1,
        figsize=(12, 8),
        sharex=True
    )   

    for i in range(latent_dim):
        axes[i].plot(z_np[i], linewidth=1)
        axes[i].set_ylabel(f"z[{i}]")
        axes[i].grid(True)  

    axes[-1].set_xlabel("Tiempo latente")   

    plt.suptitle("Dimensiones del espacio latente RAVE", fontsize=14)
    plt.tight_layout()
    plt.show()

def get_input(ruta: str) -> tuple[torch.Tensor, int]:
    x, sr = librosa.load(ruta, sr=None)
    print(f"frecuencia de muestreo {sr}")
    print(f"shape de x inicial {x.shape}")
    x = torch.from_numpy(x).reshape(1,1,-1)
    print(f"shape de x final {x.shape}")

    return x, sr

def timbre_transfer(model, x: torch.Tensor):

    z = model.encode(x)
    print(f"shape de z {z.shape} dimension que voy a modificar {z.shape[-1]}")
    z[:,0] += torch.linspace(-2,2,z.shape[-1]) #bias solo a la primera dimension

    y = model.decode(z)
    print(f"shape de y {y.shape}")

    y = y.reshape(-1)
    print("inferencia lista")

    return y, z

def sine_latent_inference(model, T_latent: float, latent_dim: int, freqs: list[float], amps: list[float]):
    t = torch.linspace(0, 1, T_latent)
    z = torch.zeros(1, latent_dim, T_latent)

    for i, (f, a) in enumerate(zip(freqs, amps)):
        z[0, i, :] = a * torch.sin(2 * np.pi * f * t)

    
    #z[0,1,:] += torch.ones(T_latent)
    y = model.decode(z)
    print(f"shape de y con latente senoidal {y.shape}")

    y = y.reshape(-1)
    print("inferencia con latente senoidal lista")

    return y, z

def user_input(input_queue):
    while True:
        pos_f, frec = input("Posicion (0-7) y frecuencia (Hz) del latente").split(",")
        pos_a, amp  =  input("Posicion (0-7) y amplitud del latente").split(",")

        update = {
                "pos_f": int(pos_f),
                "freq": float(frec),
                "pos_a": int(pos_a), 
                "amp": float(amp),
            }
        
        input_queue.put(update)

        

def play_audio(model, freqs, amps, input_queue, sr=48000):

    y = None  # audio actual

    while True:

        updated = False
        while not input_queue.empty():
            update = input_queue.get()

            freqs[update["pos_f"]] = update["freq"]
            amps[update["pos_a"]]  = update["amp"]

            updated = True

            print(">> Se actualizo la cola")
            print("freqs:", freqs)
            print("amps :", amps)

        if updated or y is None:
            y, z = sine_latent_inference(
                model,
                T_LATENTE,
                LATENT_DIM,
                freqs,
                amps
            )

            y = y.detach().cpu().numpy()

            sd.stop()
            sd.play(y, sr)

        time.sleep(0.05)


if __name__ == "__main__":

    model = torch.jit.load("percussion.ts").eval()
    print("Modelo cargado correctamente")

    x, sr_ = get_input("singing.mp3")
    y, z = timbre_transfer(model, x)
    sf.write("output_timbre_transfer.wav", y.detach().cpu().numpy(), sr_ )
    print("Archivo de audio guardado correctamente")

    """
    freqs = [150, 200, 0, 0, 0, 0, 0, 0]
    amps = [2, 1, 0, 0, 0, 0, 0, 0]

    input_queue = queue.Queue()

    hilo_input = threading.Thread(
        target=user_input,
        args=(input_queue,),
        daemon=True
    )

    hilo_audio = threading.Thread(
        target=play_audio,
        args=(model, freqs, amps, input_queue),
        daemon=True
    )

    hilo_input.start()
    hilo_audio.start()
    print("Hilos cargados correctamente")

    while True:
        time.sleep(1)
    """