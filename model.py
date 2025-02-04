import torch.nn as nn

class SoundLocalizationResNet(nn.Module):
    def __init__(self, model_type="spectra", base_architecture="resnet50", num_time_steps=4, num_channels=4, dropout_rate=0.3):
        """
        Initializes the Sound Localization ResNet model.
        
        Parameters:
        - model_type: str, "spectra" or "spectrogram"
        - base_architecture: str, "resnet34" or "resnet50"
        - num_time_steps: int, number of time steps (used for spectrogram)
        - num_channels: int, number of input channels (used for spectrogram)
        - dropout_rate: float, dropout rate for regularization
        """
        super(SoundLocalizationResNet, self).__init__()
        
        # Select base architecture
        if base_architecture == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            # self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
            self.resnet = resnet50(weights=None)
        elif base_architecture == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            # self.resnet = resnet34(weights=ResNet34_Weights.DEFAULT)
            self.resnet = resnet34(weights=None)
        else:
            raise ValueError("Invalid base_architecture. Choose 'resnet34' or 'resnet50'.")

        # Modify the first convolutional layer based on model type
        if model_type == "spectra":
            self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        elif model_type == "spectrogram":
            self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            raise ValueError("Invalid model_type. Choose 'spectra' or 'spectrogram'.")

        # Add dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)

        # Modify the final fully connected layer to output 2 values (X, Y)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            self.dropout,
            nn.Linear(num_ftrs, 2)
        )

    def forward(self, x):
        return self.resnet(x)