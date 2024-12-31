import torch.nn as nn

class CarRacingCNN1(nn.Module):
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

            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2, stride = 2),
            
            nn.Flatten(),
            nn.Linear(32 * 12 * 12, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 5)
        )
        
    def forward(self, x):
        return self.model(x)
    


    
