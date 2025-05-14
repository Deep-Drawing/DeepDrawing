import librosa
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Function to process a single audio channel and extract spectra for each frame
# def process_channel(channel_data, sample_rate, frame_rate, num_spectra_per_frame, n_fft, max_bin):
# def process_channel(channel_data, sample_rate, frame_rate, num_spectra_per_frame, n_fft):
def process_channel(channel_data, sample_rate, frame_rate, num_spectra_per_frame, n_fft, min_frequency=None, max_frequency=None):
    samples_per_frame = int(sample_rate / frame_rate)  # Calculate the number of samples per frame
    num_frames = int(len(channel_data) / samples_per_frame)  # Number of frames based on audio length
    all_frame_spectra = []  # List to store spectra for all frames

    min_bin = 0  # Default to include all frequencies from the bottom
    max_bin = None  # Default to include all frequencies to the top

    if min_frequency:
        min_bin = int(min_frequency / (sample_rate / n_fft))  # Calculate the number of FFT bins for the min frequency
    if max_frequency:
        max_bin = int(max_frequency / (sample_rate / n_fft))  # Calculate the number of FFT bins for the max frequency

    # Loop over each frame and extract spectra
    for frame in range(num_frames):
        start_sample = frame * samples_per_frame  # Calculate the start sample for the current frame
        frame_spectra = []  # List to store spectra for sub-segments within the frame

        # Divide each frame into sub-segments and calculate their spectra
        for i in range(num_spectra_per_frame):
            offset = int(i * samples_per_frame / num_spectra_per_frame)  # Calculate offset for each sub-segment
            y_segment = channel_data[start_sample + offset : start_sample + offset + samples_per_frame // num_spectra_per_frame]

            # Perform STFT to get the frequency domain representation
            D = np.abs(librosa.stft(y_segment, n_fft=n_fft, hop_length=samples_per_frame // num_spectra_per_frame))

            # Extract the last column of the STFT matrix up to max_bin (low-frequency part)
            # spectrum = np.abs(D[:max_bin, -1])  # Magnitude spectrum of the last time step in the segment
            # spectrum = np.abs(D[:, -1])  # Include all frequencies
            spectrum = np.abs(D[min_bin:max_bin, -1])  # Slice the spectrum between min_bin and max_bin
            frame_spectra.append(spectrum)  # Append the spectrum to the frame's sub-segment list

        all_frame_spectra.append(frame_spectra)  # Append all sub-segments' spectra for the current frame
    
    return all_frame_spectra  # Return the extracted spectra for all frames for this channel

# Main function to extract spectral data from multi-channel audio in a video file
# def spectraExtractor(pathToVideo, sample_rate, frame_rate, num_spectra_per_frame, n_fft, max_frequency, pathToOutputFile):
def spectraExtractor(pathToVideo, sample_rate, frame_rate, num_spectra_per_frame, n_fft, pathToOutputFile, min_frequency=None, max_frequency=None):
    # Ensure the output directory exists
    output_dir = os.path.dirname(pathToOutputFile)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load multi-channel audio from the video file at the specified sample rate
    # mono=False ensures that the audio is not down-mixed and all channels are loaded
    y, sr = librosa.load(pathToVideo, sr=sample_rate, mono=False)
    
    # Check if the audio has at least 5 channels; raise an error if not
    if y.ndim == 1 or y.shape[0] < 5:
        raise ValueError("The audio should have at least 5 channels.")
    
    # Process only the first 4 channels and skip the 5th channel (assumed to be silent)
    num_channels = 4  # Number of channels to process (ignoring the last channel)
    # audio_data = [librosa.util.normalize(y[i, :]) for i in range(num_channels)]  # Normalize audio data for each channel

    # Find the global maximum amplitude across all channels
    # global_max = np.max(np.abs(y))  # Finds the largest absolute value across all channels

    # Normalize all channels using the same global maximum
    # audio_data = [y[i, :] / (global_max + 1e-9) for i in range(num_channels)]  # Add epsilon to avoid division by zero
    audio_data = [y[i, :] for i in range(num_channels)] # No normalization

    # Calculate the number of FFT bins that correspond to the max_frequency
    # max_bin = int(max_frequency / (sample_rate / n_fft))  # Calculate the number of FFT bins for the given frequency

    # Create a partially applied function for process_channel with the fixed parameters
    # process_func = partial(process_channel, sample_rate=sample_rate, frame_rate=frame_rate, num_spectra_per_frame=num_spectra_per_frame, n_fft=n_fft, max_bin=max_bin)
    # process_func = partial(process_channel, sample_rate=sample_rate, frame_rate=frame_rate, num_spectra_per_frame=num_spectra_per_frame, n_fft=n_fft)
    process_func = partial(
        process_channel,
        sample_rate=sample_rate,
        frame_rate=frame_rate,
        num_spectra_per_frame=num_spectra_per_frame,
        n_fft=n_fft,
        min_frequency=min_frequency,  # Low-cut at 300 Hz
        max_frequency=max_frequency  # High-cut at 6400 Hz
    )

    # Process the audio channels in parallel using multiple CPU cores
    with ProcessPoolExecutor(max_workers=num_channels) as executor:
        results = list(executor.map(process_func, audio_data))  # Process each channel and collect the results

    # Combine the results into a structured format for saving
    num_frames = len(results[0])  # Number of frames should be consistent across all channels
    structured_data = []  # List to store the structured data

    # Reshape the results to organize them frame-by-frame
    for frame in range(num_frames):
        frame_data = []  # List to store data for each channel in the current frame
        for i in range(num_channels):
            frame_data.append(results[i][frame])  # Append the frame's data from each channel
        structured_data.append(frame_data)  # Append the combined data for this frame

    # Convert the structured data to a NumPy array of objects (3D array)
    structured_data = np.array(structured_data, dtype=object)
    
    # Save the structured data to an .npz file at the specified path
    np.savez(pathToOutputFile, spectra=structured_data)
    print(f'Saved structured spectra data to {pathToOutputFile}')  # Confirmation message

# # Entry point for running the script directly
# if __name__ == '__main__':
#     # Define input parameters for the function
#     filename = '2024-10-29_pen_E_01'
#     pathToVideo = f'./../Recordings/Erfun/{filename}.mp4' # Replace with your video path
#     sample_rate = 48000  # Sample rate for audio extraction (e.g., 48 kHz)
#     frame_rate = 30  # Frame rate to match video (e.g., 30 FPS)
#     num_spectra_per_frame = 3  # Number of spectra to extract per frame
#     n_fft = 2048  # FFT window size
#     max_frequency = 6400  # Maximum frequency to capture (e.g., 3000 Hz)
#     pathToOutputFile = f'{filename}.npz' # Path to save the output .npz file

#     # Run the spectraExtractor function with the provided parameters
#     # spectraExtractor(pathToVideo, sample_rate, frame_rate, num_spectra_per_frame, n_fft, max_frequency, pathToOutputFile)
#     spectraExtractor(pathToVideo, sample_rate, frame_rate, num_spectra_per_frame, n_fft, pathToOutputFile)

if __name__ == '__main__':
    # Define input parameters for the function
    filename = '2024-10-29_pen_E_01'
    pathToVideo = f'./../Recordings/Erfun/{filename}.mp4'  # Replace with your video path
    sample_rate = 48000  # Sample rate for audio extraction (e.g., 48 kHz)
    frame_rate = 30  # Frame rate to match video (e.g., 30 FPS)
    num_spectra_per_frame = 3  # Number of spectra to extract per frame
    n_fft = 2048  # FFT window size
    max_frequency = None  # Maximum frequency to capture (e.g., 6400 Hz)
    min_frequency = 200  # Minimum frequency to capture (e.g., 200 Hz)

    # Generate the output file name
    pathToOutputFile = f'{filename}_spectra_{num_spectra_per_frame}spf_{n_fft}fft_{max_frequency}hz_{frame_rate}fps.npz'

    # Run the spectraExtractor function with the provided parameters
    # spectraExtractor(pathToVideo, sample_rate, frame_rate, num_spectra_per_frame, n_fft, pathToOutputFile)
    spectraExtractor(pathToVideo, sample_rate, frame_rate, num_spectra_per_frame, n_fft, pathToOutputFile, min_frequency, max_frequency)

"""
structured_data[num_frames][num_channels][num_spectra_per_frame][max_bin]

  ├── Frame 0
  │   ├── Channel 0
  │   │   ├── Spectrum 0 (array of FFT bins up to max frequency)
  │   │   ├── Spectrum 1
  │   │   └── ...
  │   ├── Channel 1
  │   │   ├── Spectrum 0
  │   │   ├── Spectrum 1
  │   │   └── ...
  │   └── ...
  ├── Frame 1
  │   ├── Channel 0
  │   │   ├── Spectrum 0
  │   │   └── ...
  │   └── ...
  └── ...
"""




########################################################################################
########################################################################################
########################################################################################
########################################################################################

# import librosa
# import numpy as np
# import os
# from concurrent.futures import ProcessPoolExecutor
# from functools import partial
# from glob import glob

# # Function to process a single audio channel and extract spectra for each frame
# def process_channel(channel_data, sample_rate, frame_rate, num_spectra_per_frame, n_fft, max_bin):
#     samples_per_frame = int(sample_rate / frame_rate)  # Calculate the number of samples per frame
#     num_frames = int(len(channel_data) / samples_per_frame)  # Number of frames based on audio length
#     all_frame_spectra = []  # List to store spectra for all frames

#     # Loop over each frame and extract spectra
#     for frame in range(num_frames):
#         start_sample = frame * samples_per_frame  # Calculate the start sample for the current frame
#         frame_spectra = []  # List to store spectra for sub-segments within the frame

#         # Divide each frame into sub-segments and calculate their spectra
#         for i in range(num_spectra_per_frame):
#             offset = int(i * samples_per_frame / num_spectra_per_frame)  # Calculate offset for each sub-segment
#             y_segment = channel_data[start_sample + offset : start_sample + offset + samples_per_frame // num_spectra_per_frame]

#             # Perform STFT to get the frequency domain representation
#             D = np.abs(librosa.stft(y_segment, n_fft=n_fft, hop_length=samples_per_frame // num_spectra_per_frame))

#             # Extract the last column of the STFT matrix up to max_bin (low-frequency part)
#             spectrum = np.abs(D[:max_bin, -1])  # Magnitude spectrum of the last time step in the segment
#             frame_spectra.append(spectrum)  # Append the spectrum to the frame's sub-segment list

#         all_frame_spectra.append(frame_spectra)  # Append all sub-segments' spectra for the current frame
    
#     return all_frame_spectra  # Return the extracted spectra for all frames for this channel

# # Main function to extract spectral data from multi-channel audio in a session
# def spectraExtractor(session_files, sample_rate, frame_rate, num_spectra_per_frame, n_fft, max_frequency, output_file):
#     # Ensure the output directory exists
#     output_dir = os.path.dirname(output_file)
#     if output_dir and not os.path.exists(output_dir):
#         os.makedirs(output_dir)

#     # Load all channels for the session
#     num_channels = len(session_files)
#     if num_channels < 1:
#         raise ValueError(f"No channels found for session. Files: {session_files}")

#     # Normalize audio data for each channel
#     audio_data = []
#     for file in session_files:
#         y, sr = librosa.load(file, sr=sample_rate, mono=True)
#         audio_data.append(librosa.util.normalize(y))

#     # Calculate the number of FFT bins that correspond to the max_frequency
#     max_bin = int(max_frequency / (sample_rate / n_fft))  # Calculate the number of FFT bins for the given frequency

#     # Create a partially applied function for process_channel with the fixed parameters
#     process_func = partial(process_channel, sample_rate=sample_rate, frame_rate=frame_rate, 
#                            num_spectra_per_frame=num_spectra_per_frame, n_fft=n_fft, max_bin=max_bin)

#     # Process the audio channels in parallel using multiple CPU cores
#     with ProcessPoolExecutor(max_workers=num_channels) as executor:
#         results = list(executor.map(process_func, audio_data))  # Process each channel and collect the results

#     # Combine the results into a structured format for saving
#     num_frames = len(results[0])  # Number of frames should be consistent across all channels
#     structured_data = []  # List to store the structured data

#     # Reshape the results to organize them frame-by-frame
#     for frame in range(num_frames):
#         frame_data = []  # List to store data for each channel in the current frame
#         for i in range(num_channels):
#             frame_data.append(results[i][frame])  # Append the frame's data from each channel
#         structured_data.append(frame_data)  # Append the combined data for this frame

#     # Convert the structured data to a NumPy array of objects (3D array)
#     structured_data = np.array(structured_data, dtype=object)
    
#     # Save the structured data to an .npz file at the specified path
#     np.savez(output_file, spectra=structured_data)
#     print(f'Saved structured spectra data to {output_file}')  # Confirmation message

# # Entry point for running the script directly
# if __name__ == '__main__':
#     # Define dataset directory and parameters
#     dataset_dir = '../audio'  # Path to the directory containing .wav files
#     sample_rate = 48000  # Sample rate for audio extraction (e.g., 48 kHz)
#     frame_rate = 30  # Frame rate to match video (e.g., 30 FPS)
#     num_spectra_per_frame = 3  # Number of spectra to extract per frame
#     n_fft = 1024  # FFT window size
#     max_frequency = 6400  # Maximum frequency to capture (e.g., 3000 Hz)
#     output_dir = './summerSpectra'  # Directory to save .npz files

#     # Group files by session
#     all_files = glob(os.path.join(dataset_dir, '*.wav'))
#     sessions = {}
#     for file in all_files:
#         session_id = file.split('_')[1].split('-')[0]  # Extract session ID from filename
#         if session_id not in sessions:
#             sessions[session_id] = []
#         sessions[session_id].append(file)

#     # Process each session
#     for session_id, session_files in sessions.items():
#         session_files.sort()  # Ensure channels are in the correct order
#         output_file = os.path.join(output_dir, f'session_{session_id}.npz')
#         print(f"Processing session {session_id} with {len(session_files)} channels.")
#         spectraExtractor(session_files, sample_rate, frame_rate, num_spectra_per_frame, n_fft, max_frequency, output_file)