import torch
import torch.nn as nn

class AlertingCNN(nn.Module):
    def __init__(self, window_size=60):
        super(AlertingCNN, self).__init__()
        
        # 1D Convolutional Layer
        # It slides over the 60 minutes to find local patterns
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu = nn.ReLU()
        
        # Pooling reduces dimensionality and highlights important features
        self.pool = nn.MaxPool1d(kernel_size=2)
        
        # Fully connected layers to make the final decision
        # After Conv(kernel 5) -> size is 56. After Pool(2) -> size is 28.
        self.fc1 = nn.Linear(16 * 28, 32)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid() # Output between 0 and 1 (probability)

    def forward(self, x):
        # Reshape input for Conv1d: (batch, channels, length)
        # Our data is (batch, 60), we need (batch, 1, 60)
        x = x.unsqueeze(1)
        
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(x.size(0), -1) # Flatten
        x = self.relu(self.fc1(x))
        x = self.sigmoid(self.fc2(x))
        return x

if __name__ == "__main__":
    model = AlertingCNN()
    # Test with a dummy window
    dummy_input = torch.randn(8, 60) # Batch of 8 windows
    output = model(dummy_input)
    print("--- Model Test ---")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Probabilities)")