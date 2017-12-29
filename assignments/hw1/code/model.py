import torch.nn as nn


class FNN(nn.Module):
    def __init__(self,
                 input_size=11,
                 hidden1_size=200,
                 output_size=1):
        super(FNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, output_size)
        )

    def forward(self, x):
        return self.fc(x)
