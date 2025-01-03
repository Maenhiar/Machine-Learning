import torch.nn as nn

class CarRacingCNN2(nn.Module):
    """
    This class implements a CNN.
    """
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size = 5, stride = 1, padding = 2),
            nn.ReLU(),
            nn.BatchNorm2d(6),
            nn.AvgPool2d(2, stride = 2),

            nn.Conv2d(6, 16, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.AvgPool2d(2, stride = 2),
            
            nn.Flatten(),
            nn.Linear(16 * 24 * 24, 64), 
            nn.ReLU(),
            
            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        return self.model(x)