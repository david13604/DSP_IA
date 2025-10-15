from red_magenta import Autoencder

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import tensorflow as tf

os.environ["KERAS_BACKEND"] = "tensorflow"

DATA_PATH = r"C:\Users\usuario\Desktop\DSP_IA_local\Dataset_Processed"

"""
Descomentar estas lineas cuando funcione para un solo audio
all_files = sorted(
    [os.path.join(DATA_PATH, f) for f in os.listdir(DATA_PATH) if f.endswith(".npz")]
)
print(f"Se encontraron {len(all_files)} archivos de datos.")

mfccs, f0s, louds, specs = [], [], [], []


for f in tqdm(all_files, desc="Cargando dataset"):
    try:
        data = np.load(f)
        mfcc = data["mfcc"].T.astype("float32")                  # (T, 13)
        f0 = data["f0_norm"][:, np.newaxis].astype("float32")    # (T, 1)
        loud = data["loudness_db_norm"][:, np.newaxis].astype("float32")  # (T, 1)
        stft_mag = data["stft_mag"].astype("float32")            # (F, T)

        # Guardar
        mfccs.append(mfcc)
        f0s.append(f0)
        louds.append(loud)
        specs.append(stft_mag)
    except Exception as e:
        print(f"Error cargando {f}: {e}")


T_min = min([m.shape[0] for m in mfccs])
F_min = min([s.shape[0] for s in specs])
print(f"Longitud temporal mínima: {T_min}, bins espectrales mínimos: {F_min}")

for i in range(len(mfccs)):
    mfccs[i] = mfccs[i][:T_min, :]
    f0s[i] = f0s[i][:T_min, :]
    louds[i] = louds[i][:T_min, :]
    specs[i] = specs[i][:F_min, :]


mfcc_all = np.stack(mfccs)     # (B, T, 13)
f0_all = np.stack(f0s)         # (B, T, 1)
loud_all = np.stack(louds)     # (B, T, 1)
spec_all = np.stack(specs)     # (B, F, Tstft)

print("Dataset shapes:")
print("  mfcc_all:", mfcc_all.shape)
print("  f0_all:", f0_all.shape)
print("  loud_all:", loud_all.shape)
print("  spec_all:", spec_all.shape)
"""

data = np.load('SampLib_DPA_138_Seg0.npz')
print('si')
mfcc = data["mfcc"].T.astype("float32")[np.newaxis, ...]      # (1, 250, 13)
f0 = data["f0_norm"][:, np.newaxis].astype("float32")[np.newaxis, ...]  # (1, 250, 1)
loud = data["loudness_db_norm"][:, np.newaxis].astype("float32")[np.newaxis, ...]  # (1, 250, 1)
stft_mag = data["stft_mag"].astype("float32")[np.newaxis, ...]  # (1, 1025, 251)

f0_hz_true = data["f0_hz"][:, np.newaxis].astype("float32")[np.newaxis, ...]
loud_lin_true = data["loudness_lin"][:, np.newaxis].astype("float32")[np.newaxis, ...]

# prints para los shapes
print("mfcc:", mfcc.shape)
print("f0:", f0.shape)
print("loud:", loud.shape)
print("stft:", stft_mag.shape)

model = Autoencder(
    input_shape=(None, 13),
    z_dim=16,
    n_harmonics=101,
    sample_rate=16000,
    n_samples=64000,
    stft_frame_length=2048,
    stft_frame_step=256
)
model.f0_real = tf.constant(f0_hz_true, dtype=tf.float32)
model.loud_real = tf.constant(loud_lin_true, dtype=tf.float32)
auto = model.build()
model.compile(lr=1e-4)

history = model.fit(
    x_train=[mfcc, f0, loud],
    y_train=stft_mag,
    batch_size=1,
    epochs=200
)

plt.plot(history.history["loss"], label="train loss")
plt.legend()
plt.show()

model.autoencoder.save("ddsp_autoencoder.h5")