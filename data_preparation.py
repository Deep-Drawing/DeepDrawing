import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from PIL import Image

def load_csv_data(csv_file):
    """Load CSV data and print basic info."""
    df = pd.read_csv(csv_file)
    print(f"Loaded CSV file with {len(df)} rows from {csv_file}")
    return df

def load_spectra(npz_file):
    """Load spectra data from .npz file."""
    try:
        data = np.load(npz_file, allow_pickle=True)
        spectra = data['spectra']
        print(f"Loaded spectra from {npz_file}. Shape: {spectra.shape}")
        return spectra
    except FileNotFoundError:
        print(f"Error: File not found - {npz_file}")
        return None

def prepare_data(data_type="spectrogram", normalize=True, csv_dir=None, data_dir=None):
    """
    Unified function to prepare data for both spectrograms and spectra.
    """
    if csv_dir is None or data_dir is None:
        raise ValueError("Both `csv_dir` and `data_dir` must be specified.")
    
    metadata = []  # Store metadata for streaming or spectra
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files in '{csv_dir}'.")
    
    spectra_cache = {}  # Cache to store loaded spectra per .npz file

    for csv_file in csv_files:
        dataset_name = os.path.splitext(csv_file)[0]
        csv_path = os.path.join(csv_dir, csv_file)

        # Load CSV
        df = load_csv_data(csv_path)

        # Filter rows where Sound is True
        df = df[df['Sound'] == True]
        print(f"Filtered rows with sound. Remaining rows: {len(df)}")

        # Further filter out rows where both X and Y are absolute 0
        df = df[~((df['X'] == 0) & (df['Y'] == 0))]
        print(f"Removed rows where X and Y are both 0. Remaining rows: {len(df)}")

        # Load spectra for this dataset (only once)
        if data_type == "spectra":
            spectra_path = os.path.join(data_dir, f"{dataset_name}.npz")
            if dataset_name not in spectra_cache:
                spectra_data = load_spectra(spectra_path)
                if spectra_data is None:
                    print(f"Skipping {dataset_name} due to missing spectra.")
                    continue
                spectra_cache[dataset_name] = spectra_data  # Cache the loaded spectra
            else:
                spectra_data = spectra_cache[dataset_name]

        for index, row in df.iterrows():
            frame = int(row['Frame'])

            if data_type == "spectrogram":
                # Prepare spectrogram paths for all channels
                spectrogram_paths = {}
                missing_spectrogram = False
                for channel in range(1, 5):  # Channels 1 to 4
                    channel_dir = os.path.join(data_dir, f"{dataset_name}_spectrogram", f"channel_{channel}")
                    image_path = os.path.join(channel_dir, f"spectrogram_{frame:04d}.png")
                    if os.path.exists(image_path):
                        spectrogram_paths[f"channel_{channel}"] = image_path
                    else:
                        print(f"Missing spectrogram for channel {channel}, frame {frame}. Skipping frame.")
                        missing_spectrogram = True
                        break
                
                if missing_spectrogram:
                    continue  # Skip this frame if any spectrogram is missing

                # Add metadata for spectrograms
                metadata.append({
                    "dataset_name": dataset_name,
                    "frame": frame,
                    "X": row['X'],
                    "Y": row['Y'],
                    "spectrogram_paths": spectrogram_paths
                })

            elif data_type == "spectra":
                if frame >= len(spectra_data):
                    print(f"Frame {frame} exceeds data length for {dataset_name}. Skipping.")
                    continue
                
                # Add metadata for spectra
                metadata.append({
                    "spectra": spectra_data[frame],
                    "X": row["X"],
                    "Y": row["Y"]
                })

    print(f"Collected metadata for {len(metadata)} frames.")

    # Normalize X and Y coordinates
    if normalize:
        print("Normalizing coordinates using fixed bounds (0 and 936)...")
        for item in metadata:
            item['X'] = item['X'] / 936
            item['Y'] = item['Y'] / 936

    # Split metadata into train, validation, and test
    np.random.seed(42)
    np.random.shuffle(metadata)

    num_samples = len(metadata)
    train_split = int(0.7 * num_samples)
    val_split = int(0.85 * num_samples)

    train_metadata = metadata[:train_split]
    val_metadata = metadata[train_split:val_split]
    test_metadata = metadata[val_split:]

    if data_type == "spectrogram":
        print(f"Train size: {len(train_metadata)}, Val size: {len(val_metadata)}, Test size: {len(test_metadata)}")
        return train_metadata, val_metadata, test_metadata

    elif data_type == "spectra":
        # Convert metadata to preloaded arrays
        X_train = np.array([item['spectra'] for item in train_metadata], dtype=object)
        y_train = np.array([{"X": item["X"], "Y": item["Y"]} for item in train_metadata], dtype=object)
        X_val = np.array([item['spectra'] for item in val_metadata], dtype=object)
        y_val = np.array([{"X": item["X"], "Y": item["Y"]} for item in val_metadata], dtype=object)
        X_test = np.array([item['spectra'] for item in test_metadata], dtype=object)
        y_test = np.array([{"X": item["X"], "Y": item["Y"]} for item in test_metadata], dtype=object)

        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    print("Starting data preparation...")

    # Toggle between "spectrogram" and "spectra"
    data_type = input("Enter data type ('spectrogram' or 'spectra'): ").strip().lower()
    if data_type not in ["spectrogram", "spectra"]:
        raise ValueError("Invalid data type. Please enter 'spectrogram' or 'spectra'.")

    # Specify whether to normalize
    normalize_input = input("Normalize data? (Y/N): ").strip().lower()
    normalize = normalize_input == "y"

    # Specify paths for CSV and data directories
    csv_dir = input("Enter the path to the CSV directory (default: '../Dataset/Coordinates'): ").strip()
    if csv_dir == "":
        csv_dir = "../Dataset/Coordinates"
    
    data_dir = input(f"Enter the path to the data directory (default: '../Dataset/{'Spectra' if data_type == 'spectra' else 'Spectrograms'}'): ").strip()
    if data_dir == "":
        data_dir = f"../Dataset/{'Spectra' if data_type == 'spectra' else 'Spectrograms'}"

    # Optional custom naming
    custom_name = input("You can enter an arbitrary name if you want. If not, just skip by pressing Enter: ").strip()
    user_name = f"{custom_name}_" if custom_name else ""

    # Prepare data
    print("Preparing data...")
    if data_type == "spectrogram":
        train_metadata, val_metadata, test_metadata = prepare_data(
            data_type=data_type, 
            normalize=normalize, 
            csv_dir=csv_dir, 
            data_dir=data_dir
        )

        # Determine output file name
        if normalize:
            output_file = f'dataset/{user_name}processed_data_normalized_{data_type}.npz'
        else:
            output_file = f'dataset/{user_name}processed_data_{data_type}.npz'

        # Save streaming metadata
        print("Saving metadata for streaming...")
        np.savez(output_file, 
                 train=train_metadata, 
                 val=val_metadata, 
                 test=test_metadata)
        print(f"Metadata saved successfully to {output_file}!")

    elif data_type == "spectra":
        X_train, X_val, X_test, y_train, y_val, y_test = prepare_data(
            data_type=data_type, 
            normalize=normalize, 
            csv_dir=csv_dir, 
            data_dir=data_dir
        )
        # Determine output file name
        if normalize:
            output_file = f'dataset/{user_name}processed_data_normalized_{data_type}.npz'
        else:
            output_file = f'dataset/{user_name}processed_data_{data_type}.npz'

        # Save processed data
        print("Saving processed data...")
        np.savez(output_file, 
                 X_train=X_train, X_val=X_val, X_test=X_test, 
                 y_train=y_train, y_val=y_val, y_test=y_test)
        print(f"Data saved successfully to {output_file}!")

