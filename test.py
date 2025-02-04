import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from PIL import Image
from model import SoundLocalizationResNet


class SpectrogramDataset(Dataset):
    """Dataset class for spectrograms."""
    def __init__(self, metadata, data_dir):
        self.metadata = metadata
        self.data_dir = data_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        dataset_name = item["dataset_name"]
        frame = item["frame"]

        # Load spectrograms for 4 channels
        spectrograms = []
        for channel in range(1, 5):  # Channels 1 to 4
            spectrogram_path = os.path.join(
                self.data_dir,
                f"{dataset_name}_spectrogram",
                f"channel_{channel}",
                f"spectrogram_{str(frame).zfill(4)}.png"
            )
            with Image.open(spectrogram_path) as img:
                spectrogram = np.array(img.convert("L"))  # Convert to grayscale
                spectrograms.append(spectrogram)

        # Combine all channels into a single tensor
        combined_spectrogram = np.stack(spectrograms, axis=2)  # Shape: (H, W, Channels)
        combined_spectrogram = combined_spectrogram.transpose(2, 0, 1)  # Shape: (Channels, H, W)

        # Convert to tensor
        spectrogram_tensor = torch.tensor(combined_spectrogram, dtype=torch.float32)

        # Get normalized coordinates
        X, Y = item["X"], item["Y"]
        coordinates_tensor = torch.tensor([X, Y], dtype=torch.float32)

        return spectrogram_tensor, coordinates_tensor


def prepare_spectra_data(X, y):
    """Prepare spectra data for testing."""
    X_list = []
    for i, item in enumerate(X):
        reshaped_spectra = np.concatenate(
            [item[channel] for channel in range(len(item))], axis=0
        )  # Combine sub-segments as virtual channels
        X_list.append(reshaped_spectra)

    X_tensor = torch.tensor(np.array(X_list, dtype=np.float32))
    X_tensor = X_tensor.unsqueeze(1)  # Add height=1 dimension for ResNet (N, C, H, W)

    y_tensor = torch.tensor([[item['X'], item['Y']] for item in y], dtype=torch.float32)

    return X_tensor, y_tensor


def evaluate_model(data_type, model, test_loader, device):
    """Evaluate the model on the test dataset."""
    model.eval()
    total_loss = 0.0
    criterion = torch.nn.L1Loss()  # Adjust criterion if needed

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader)
    return avg_loss


if __name__ == "__main__":
    # User-defined options
    data_type = input("Enter data type (spectra/spectrogram): ").strip().lower()
    model_path = input("Enter path to the saved model: ").strip()
    data_dir = "../Dataset/Spectrograms" if data_type == "spectrogram" else "../Dataset/Spectra_Normalized_noHighCut"

    # Let the user choose the dataset file
    dataset_folder = "dataset"
    available_files = [f for f in os.listdir(dataset_folder) if f.endswith('.npz')]

    if not available_files:
        raise FileNotFoundError("No processed dataset files found in the 'dataset' folder.")

    print("\nAvailable dataset files:")
    for idx, file in enumerate(available_files):
        print(f"{idx}: {file}")

    file_index = int(input(f"Select the file index (0-{len(available_files)-1}): ").strip())
    dataset_path = os.path.join(dataset_folder, available_files[file_index])

    # Load test data
    print("Loading test data...")
    # Load selected dataset
    data = np.load(dataset_path, allow_pickle=True)

    # Prepare datasets and loaders
    if data_type == "spectrogram":
        test_metadata = data['test']  # Use metadata for streaming
        test_dataset = SpectrogramDataset(test_metadata, data_dir)
    elif data_type == "spectra":
        X_test = data['X_test']
        y_test = data['y_test']
        X_test_tensor, y_test_tensor = prepare_spectra_data(X_test, y_test)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the model
    model = SoundLocalizationResNet(
        model_type=data_type,
        base_architecture="resnet50"  # Change to "resnet34" if needed
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    # Evaluate the model
    print("Evaluating the model on the test set...")
    avg_test_loss = evaluate_model(data_type, model, test_loader, device)
    print(f"Test Loss: {avg_test_loss:.4f}")