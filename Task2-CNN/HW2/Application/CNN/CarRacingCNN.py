import torch.nn as nn

class CarRacingCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Definizione della rete usando nn.Sequential
        self.model = nn.Sequential(
            # Convolutional Layer 1: 6 filters, 5x5 kernel
            nn.Conv2d(3, 6, kernel_size = 5),  # Input: 3x96x96, Output: 6x92x92
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),  # Output: 6x46x46
            
            # Convolutional Layer 2: 16 filters, 5x5 kernel
            nn.Conv2d(6, 16, kernel_size = 5),  # Output: 16x42x42
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),  # Output: 16x21x21
            
            # Fully connected layer (Flatten the 3D tensor into 1D)
            nn.Flatten(),
            nn.Linear(16 * 21 * 21, 120),  # Output: 120
            nn.ReLU(),
            nn.Linear(120, 84),  # Output: 84
            nn.ReLU(),
            nn.Linear(84, 5)  # Output: 5 (for 5 classes, for example)
        )
        
    def forward(self, x):
        return self.model(x)
