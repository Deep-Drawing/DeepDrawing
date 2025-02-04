import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
from datetime import datetime
from PIL import Image
from model import SoundLocalizationResNet  # Unified model

def log_message(message, log_file_path=None):
    """Logs messages to the console and optionally to a file."""
    if log_file_path:
        with open(log_file_path, "a") as log_file:
            log_file.write(message + "\n")
    print(message)


class SpectrogramDataset(Dataset):
    """Dataset class for spectrograms."""
    def __init__(self, metadata, data_dir, transform=None):
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transform

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
    """Prepares spectra data for training."""
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


def train_model(data_type, model, train_loader, val_loader, optimizer, scheduler, criterion, device,
                num_epochs, patience, use_early_stopping, weights_folder, log_file_path):
    """Handles the training loop."""
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        log_message(f"Epoch {epoch}: Training Loss = {avg_train_loss:.4f}", log_file_path)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Val]"):
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        log_message(f"Epoch {epoch}: Validation Loss = {avg_val_loss:.4f}", log_file_path)

        # Save weights for the current epoch
        torch.save(model.state_dict(), os.path.join(weights_folder, f"model_epoch_{epoch}.pth"))

        # Early stopping logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if use_early_stopping and epochs_no_improve >= patience:
            log_message(f"Early stopping triggered after {patience} epochs without improvement.", log_file_path)
            break

    log_message(f"Training completed. Best Validation Loss: {best_val_loss:.4f}", log_file_path)


if __name__ == "__main__":
    # User-defined options
    data_type = input("Enter data type (spectra/spectrogram): ").strip().lower()
    experiment_name = input("Enter experiment name: ").strip()
    use_early_stopping = input("Enable early stopping? (Y/N): ").strip().lower() == 'y'
    patience = int(input("Enter patience for early stopping (default: 20): ") or 20)

    # Experiment setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_folder = f"trained_models/{experiment_name}_{timestamp}"
    os.makedirs(experiment_folder, exist_ok=True)
    weights_folder = os.path.join(experiment_folder, "weights")
    os.makedirs(weights_folder, exist_ok=True)
    log_file_path = os.path.join(experiment_folder, "training_log.txt")

    log_message("Loading data...", log_file_path)
    # data = np.load(f'dataset/processed_data_normalized_{data_type}.npz', allow_pickle=True)
    dataset_folder = "dataset"
    print("Available preprocessed files:")
    available_files = [f for f in os.listdir(dataset_folder) if f.endswith(".npz")]
    for idx, file in enumerate(available_files):
        print(f"{idx}: {file}")

    while True:
        try:
            file_index = int(input(f"Select the file index (0-{len(available_files) - 1}): "))
            if 0 <= file_index < len(available_files):
                selected_file = available_files[file_index]
                break
            else:
                print("Invalid index. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    data = np.load(os.path.join(dataset_folder, selected_file), allow_pickle=True)

    # X_train, X_val = data['X_train'], data['X_val']
    # y_train, y_val = data['y_train'], data['y_val']

    if data_type == "spectra":
        X_train, X_val = data['X_train'], data['X_val']
        y_train, y_val = data['y_train'], data['y_val']
    elif data_type == "spectrogram":
        train_metadata, val_metadata = data['train'], data['val']

    # Prepare datasets and loaders
    if data_type == "spectrogram":
        train_dataset = SpectrogramDataset(train_metadata, "../Dataset/Spectrograms")
        val_dataset = SpectrogramDataset(val_metadata, "../Dataset/Spectrograms")
    elif data_type == "spectra":
        X_train_tensor, y_train_tensor = prepare_spectra_data(X_train, y_train)
        X_val_tensor, y_val_tensor = prepare_spectra_data(X_val, y_val)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize the unified model
    model = SoundLocalizationResNet(
        model_type=data_type,
        base_architecture="resnet50"  # Change to "resnet34" if needed
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = OneCycleLR(optimizer, max_lr=1e-3, epochs=120, steps_per_epoch=len(train_loader))
    criterion = torch.nn.L1Loss()

    # Train the model
    train_model(data_type, model, train_loader, val_loader, optimizer, scheduler, criterion,
                torch.device("cuda" if torch.cuda.is_available() else "cpu"), 120, patience,
                use_early_stopping, weights_folder, log_file_path)