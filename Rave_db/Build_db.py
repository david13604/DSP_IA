import os
import soundfile as sf
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from pathlib import Path

# 48kHz audio synthesis
TARGET_SAMPLE_RATE = 48000 
# train/test split
TEST_SIZE = 0.10


SOURCE_DATA_DIR = "/mnt/c/Users/matth/Desktop/Other/DSP_IA/Rave_db/BuK_all" 
# Where the ready-to-train database will be saved
OUTPUT_DB_DIR = "/mnt/c/Users/matth/Desktop/Other/DSP_IA/Rave_db/RAVE_db"

def preprocess_audio(file_path, output_path):
    try:
        data, samplerate = sf.read(file_path)

        # Convert to Mono
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample to 48kHz (RAVE requirement)
        if samplerate != TARGET_SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=samplerate, target_sr=TARGET_SAMPLE_RATE)

        # Normalization
        max_val = np.abs(data).max()
        if max_val > 0:
            data = data / max_val

        # Save
        sf.write(output_path, data, TARGET_SAMPLE_RATE, subtype='PCM_16')
        return True

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def generate_database():
    # Create output directories
    train_dir = Path(OUTPUT_DB_DIR) / "train"
    test_dir = Path(OUTPUT_DB_DIR) / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning files in {SOURCE_DATA_DIR}...")
    all_files = list(Path(SOURCE_DATA_DIR).rglob("*.wav"))
    
    if len(all_files) == 0:
        print("No .wav files found! Check your SOURCE_DATA_DIR.")
        return

    print(f"Found {len(all_files)} audio files.")

    # Split train/test
    train_files, test_files = train_test_split(all_files, test_size=TEST_SIZE, random_state=42)

    print(f"Processing Training Set ({len(train_files)} files)...")
    for file_path in tqdm(train_files):
        out_name = f"{file_path.stem}_48k.wav"
        preprocess_audio(str(file_path), str(train_dir / out_name))

    print(f"Processing Test Set ({len(test_files)} files)...")
    for file_path in tqdm(test_files):
        out_name = f"{file_path.stem}_48k.wav"
        preprocess_audio(str(file_path), str(test_dir / out_name))

    print("\n--- Database Generation Complete ---")
    print(f"Dataset prepared at: {OUTPUT_DB_DIR}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DATA_DIR):
        print("Please edit the script to point to your downloaded dataset.")
    else:
        generate_database()