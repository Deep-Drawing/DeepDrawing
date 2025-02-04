import os
import numpy as np
import torch
import librosa
import asyncio
import websockets
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
import io
from model import SoundLocalizationResNet
from prediction_filter import PredictionFilter
import sounddevice as sd
import time

print(sd.query_devices())

# ==============================
# User Configurable Parameters
# ==============================

MODEL_TYPE = input("Enter model type ('spectra' or 'spectrogram'): ").strip().lower()
MODEL_PATH = input("Enter path to trained model (.pth file): ").strip()

# Audio Parameters
SAMPLE_RATE = 48000
WINDOW_LENGTH = 1600
# WINDOW_LENGTH = 1024
HOP_LENGTH = 512
BUFFER_DURATION = 5  # seconds
NUM_CHANNELS = 4
SOUND_THRESHOLD = 0.0005  # Threshold for detecting sound
TIME_STEPS = 4  # Used for spectrogram processing
PREDICTION_LOG_FILE = "prediction_speed_log.txt"
PREDICTION_COUNTER_INTERVAL = 5  # Log predictions per second every 5s
USE_HIGH_CUT = input("Did the trained model use a high-cut (6400 Hz)? (yes/no): ").strip().lower() == "yes"

# Data Directories
# if MODEL_TYPE == "spectrogram":
#     DATA_DIR = "Dataset/Spectrograms"
#     SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT = 310, 308  # Expected dimensions
# elif MODEL_TYPE == "spectra":
#     DATA_DIR = "Dataset/Spectra_Normalized_noHighCut"
# else:
#     raise ValueError("Invalid model type. Choose 'spectra' or 'spectrogram'.")

# ==============================
# Load Model
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SoundLocalizationResNet(model_type=MODEL_TYPE).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

print(f"Loaded {MODEL_TYPE} model from {MODEL_PATH}")

# ==============================
# Initialize Audio Processing
# ==============================

audio_buffers = np.zeros((NUM_CHANNELS, int(BUFFER_DURATION * SAMPLE_RATE)))
initial_state = np.array([468, 468])
pred_filter = PredictionFilter(initial_state, window_size=5, distance_threshold=200, warm_up_count=10)

# ==============================
# Setup for Visualization
# ==============================

plt.ion()
fig, ax = plt.subplots(figsize=(8, 8))
scatter = ax.scatter([], [], c="red", label="Predicted Positions")
ax.set_xlim(0, 1000)
ax.set_ylim(1000, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Real-time Sound Source Localization")
ax.legend()

recent_predictions = []
MAX_PREDICTIONS = 50  # Number of recent predictions to display
prediction_count = 0
last_counter_reset = time.time()

# ==============================
# Helper Functions
# ==============================

def update_plot(cx, cy):
    """Update the visualization plot."""
    global recent_predictions
    recent_predictions.append((cx, cy))
    if len(recent_predictions) > MAX_PREDICTIONS:
        recent_predictions.pop(0)

    x, y = zip(*recent_predictions)
    scatter.set_offsets(np.c_[x, y])
    fig.canvas.draw_idle()
    plt.pause(0.01)

def extract_spectrogram(audio_buffer):
    """Compute and return a Mel spectrogram."""
    D = np.abs(librosa.stft(audio_buffer, n_fft=2048, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH))
    # S_db = librosa.amplitude_to_db(D, ref=np.max)
    return D

def extract_spectra(audio_buffer):
    """Compute and return a magnitude spectrum (STFT) for spectra processing."""
    D = np.abs(librosa.stft(audio_buffer, n_fft=1024, hop_length=HOP_LENGTH, win_length=WINDOW_LENGTH))
    return D[:6400, :] if USE_HIGH_CUT else D  # Apply high-cut only if enabled

def is_sound_detected(audio_buffers):
    """Check if sound is detected based on RMS values."""
    rms_values = [np.mean(librosa.feature.rms(y=buffer)) for buffer in audio_buffers]
    return any(rms > SOUND_THRESHOLD for rms in rms_values), rms_values

def log_prediction_speed():
    """Log the speed of predictions per second."""
    global prediction_count, last_counter_reset
    current_time = time.time()
    elapsed_time = current_time - last_counter_reset

    if elapsed_time >= PREDICTION_COUNTER_INTERVAL:
        predictions_per_second = prediction_count / elapsed_time
        with open(PREDICTION_LOG_FILE, "a") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp}: {predictions_per_second:.2f} predictions/second\n")

        prediction_count = 0
        last_counter_reset = current_time

# ==============================
# WebSocket Server
# ==============================

async def send_coordinates(websocket, path):
    """Send real-time sound localization coordinates via WebSocket."""
    global prediction_count  # Add this line

    while True:
        sound_detected, rms_values = is_sound_detected(audio_buffers)

        if sound_detected:
            # Extract features
            if MODEL_TYPE == "spectrogram":
                spectrograms = [extract_spectrogram(audio_buffers[ch, -WINDOW_LENGTH:]) for ch in range(NUM_CHANNELS)]
                combined_input = np.stack(spectrograms, axis=2)
                input_tensor = torch.tensor(combined_input, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)

            elif MODEL_TYPE == "spectra":
                spectra = [extract_spectra(audio_buffers[ch, -WINDOW_LENGTH:]) for ch in range(NUM_CHANNELS)]
                combined_input = np.concatenate(spectra, axis=1)
                input_tensor = torch.tensor(combined_input, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

            # Send to model
            input_tensor = input_tensor.to(device)
            with torch.no_grad():
                prediction = model(input_tensor)

            cx, cy = prediction.cpu().numpy().flatten()

            # Apply filtering
            filtered_prediction = pred_filter.update(np.array([cx, cy]))
            if filtered_prediction is not None:
                cx, cy = filtered_prediction
                update_plot(cx, cy)
                await websocket.send(f"{int(cx)},{int(cy)}, 0")

            prediction_count += 1  # No more UnboundLocalError
            log_prediction_speed()

        else:
            print("No sound detected.")

        await asyncio.sleep(0.01)

# ==============================
# Main Execution
# ==============================

def callback(indata, frames, time, status):
    """Update audio buffer for real-time processing."""
    global audio_buffers
    audio_buffers[:, :-frames] = audio_buffers[:, frames:]
    audio_buffers[:, -frames:] = indata.T  # Update with new audio data

async def main():
    print("Starting WebSocket server...")
    async with websockets.serve(send_coordinates, "localhost", 8765):
        print("WebSocket server started")
        await asyncio.Future()  # run forever

if __name__ == "__main__":
    with sd.InputStream(callback=callback,
                    channels=NUM_CHANNELS,
                    samplerate=SAMPLE_RATE,
                    blocksize=WINDOW_LENGTH):
        print("Streaming live audio... Press Ctrl+C to stop.")
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("Stopped live audio streaming.")
        except Exception as e:
            print(f"Error: {e}")