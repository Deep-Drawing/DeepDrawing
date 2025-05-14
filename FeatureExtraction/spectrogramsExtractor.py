import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.use('Agg')  # Use Agg backend for non-interactive plotting

def print_colored(message, color):
    """Function to print colored messages."""
    color_code = {
        'green': '\033[92m',
        'red': '\033[91m',
        'blue': '\033[94m',
        'end': '\033[0m'
    }
    print(f"{color_code[color]}{message}{color_code['end']}")

# def process_recording(file_path, output_base_dir, video_fps=30, n_fft=2048, hop_length=512, win_length=1600):
#     """
#     Processes a single multi-channel video file, extracts spectrograms, 
#     and saves them in the specified output directory structure.
#     Skips the 5th (mute) channel.
#     """
#     recording_name = os.path.splitext(os.path.basename(file_path))[0]
#     output_dir = os.path.join(output_base_dir, f"{recording_name}_spectrogram")
#     os.makedirs(output_dir, exist_ok=True)

#     print_colored(f"Processing {recording_name}...", "blue")

#     # Load multi-channel audio
#     y, sr = librosa.load(file_path, sr=None, mono=False)  # Load multi-channel audio
#     num_channels = y.shape[0] if y.ndim == 2 else 1

#     if num_channels < 5:
#         print_colored(f"Warning: {recording_name} has less than 5 channels! Found {num_channels} channels.", "red")

#     print(f"Sample rate: {sr}, Number of channels: {num_channels} (processing first 4 channels)")

#     # Frame parameters
#     frame_duration = 1.0 / video_fps
#     total_frames = int(librosa.get_duration(y=y[0], sr=sr) * video_fps)

#     # Loop through the first 4 channels only
#     for ch in range(min(4, num_channels)):
#         channel_dir = os.path.join(output_dir, f"channel_{ch + 1}")
#         os.makedirs(channel_dir, exist_ok=True)
#         print_colored(f"  Processing Channel {ch + 1}", "green")

#         # Process each frame
#         for frame in range(total_frames):
#             output_path = os.path.join(channel_dir, f"spectrogram_{frame + 1:04d}.png")

#             # Check if the spectrogram already exists
#             if os.path.exists(output_path):
#                 print(f"    Skipping existing spectrogram: {output_path}")
#                 continue

#             # Determine start and end samples for this frame
#             start_sample = int(frame * frame_duration * sr)
#             end_sample = int((frame + 1) * frame_duration * sr)

#             # Extract audio for this frame
#             y_frame = y[ch, start_sample:end_sample]

#             # Compute STFT
#             D = np.abs(librosa.stft(y_frame, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
#             S_db = librosa.amplitude_to_db(D, ref=np.max)

#             # Save the spectrogram
#             plt.figure(figsize=(4, 4))
#             plt.axis('off')
#             librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
#             plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#             plt.close()

#             print(f"    Saved spectrogram for Channel {ch + 1}, Frame {frame + 1}/{total_frames}")

#     print_colored(f"Completed processing {recording_name}!", "green")

def process_recording(file_path, output_base_dir, video_fps=30, n_fft=2048, hop_length=512, win_length=1600):
    """
    Processes a single multi-channel video file, extracts spectrograms, 
    and saves them in the specified output directory structure.
    Skips the 5th (mute) channel.
    """
    recording_name = os.path.splitext(os.path.basename(file_path))[0]
    output_dir = os.path.join(output_base_dir, f"{recording_name}_spectrogram")
    os.makedirs(output_dir, exist_ok=True)

    print_colored(f"Processing {recording_name}...", "blue")

    # Load multi-channel audio
    y, sr = librosa.load(file_path, sr=None, mono=False)  # Load multi-channel audio
    num_channels = y.shape[0] if y.ndim == 2 else 1

    if num_channels < 5:
        print_colored(f"Warning: {recording_name} has less than 5 channels! Found {num_channels} channels.", "red")

    print(f"Sample rate: {sr}, Number of channels: {num_channels} (processing first 4 channels)")

    # Global normalization: Find the maximum amplitude across all channel
    # global_max = np.max(np.abs(y))

    # Normalize the entire audio by the global maximum amplitude
    # y = y / (global_max + 1e-9)  # Add epsilon to avoid division by zero

    # Frame parameters
    frame_duration = 1.0 / video_fps
    total_frames = int(librosa.get_duration(y=y[0], sr=sr) * video_fps)

    # Loop through the first 4 channels only
    for ch in range(min(4, num_channels)):
        channel_dir = os.path.join(output_dir, f"channel_{ch + 1}")
        os.makedirs(channel_dir, exist_ok=True)
        print_colored(f"  Processing Channel {ch + 1}", "green")

        # Process each frame
        for frame in range(total_frames):
            output_path = os.path.join(channel_dir, f"spectrogram_{frame + 1:04d}.png")

            # Check if the spectrogram already exists
            if os.path.exists(output_path):
                print(f"    Skipping existing spectrogram: {output_path}")
                continue

            # Determine start and end samples for this frame
            start_sample = int(frame * frame_duration * sr)
            end_sample = int((frame + 1) * frame_duration * sr)

            # Extract audio for this frame
            y_frame = y[ch, start_sample:end_sample]

            # Compute STFT
            # D = np.abs(librosa.stft(y_frame, n_fft=n_fft, hop_length=hop_length, win_length=win_length))

            mel_spec = librosa.feature.melspectrogram(
                y=y_frame,
                sr=sr,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                n_mels=128  # Number of Mel bands
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)  # Convert to decibels (optional)

            # Save the spectrogram (without amplitude normalization)
            plt.figure(figsize=(4, 4))
            plt.axis('off')
            # librosa.display.specshow(D, sr=sr, hop_length=hop_length, x_axis=None, y_axis=None)
            librosa.display.specshow(mel_spec_db, sr=sr, hop_length=hop_length, x_axis=None, y_axis='mel')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            print(f"    Saved spectrogram for Channel {ch + 1}, Frame {frame + 1}/{total_frames}")

    print_colored(f"Completed processing {recording_name}!", "green")

def batch_process_recordings(input_dir, output_dir, video_fps=30):
    """
    Processes all .mp4 files in the input directory and its subdirectories.
    Extracts spectrograms and saves them to the output directory.
    Skips already existing spectrograms to avoid recomputation.
    """
    print_colored(f"Scanning directory: {input_dir}", "blue")

    # Walk through the directory and find all .mp4 files
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".mp4"):
                file_path = os.path.join(root, file)
                process_recording(file_path, output_dir, video_fps=video_fps)

    print_colored("Batch processing complete!", "green")


if __name__ == "__main__":
    # Input directory containing the recordings
    input_directory = "../PenRecordings"  # Replace with your recordings directory
    output_directory = "../MelSpectrograms_noNormalization"  # Where the spectrograms will be saved

    # Parameters
    video_fps = 30  # Adjust if needed

    # Run batch processing
    batch_process_recordings(input_directory, output_directory, video_fps=video_fps)