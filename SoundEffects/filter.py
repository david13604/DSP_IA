import os
import shutil
from pydub import AudioSegment

AUDIO_EXTENSIONS = ('.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac')
MAX_DURATION_SECONDS = 4
TARGET_SAMPLE_RATE = 44100
TARGET_CHANNELS = 1
TARGET_DURATION = 4

def process_audio_file(filepath):
    try:
        audio = AudioSegment.from_file(filepath)
        duration = len(audio) / 1000.0  # duration in seconds
        if duration > MAX_DURATION_SECONDS:
            print(f"Deleting {filepath} (duration: {duration:.2f}s)")
            os.remove(filepath)
            return
        elif duration < TARGET_DURATION:
            print(f"Padding {filepath} (duration: {duration:.2f}s)")
            silence = AudioSegment.silent(duration=(TARGET_DURATION - duration) * 1000)
            audio = audio + silence

        # Standardize audio
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE).set_channels(TARGET_CHANNELS)
        wav_path = os.path.splitext(filepath)[0] + '.wav'
        audio.export(wav_path, format='wav')
        if filepath != wav_path:
            os.remove(filepath)
        print(f"Standardized {wav_path}")
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

def process_folder(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(AUDIO_EXTENSIONS):
                filepath = os.path.join(dirpath, filename)
                process_audio_file(filepath)

# Run the function directly, allowing folder input
import sys
folder = "/mnt/c/SoundEffects"
process_folder(folder)