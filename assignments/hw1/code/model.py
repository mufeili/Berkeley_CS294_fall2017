import torch.nn as nn


class FNN(nn.Module):

    def __init__(self, in_size=11, hidden1_size=200, out_size=1):
        super(FNN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(in_size, hidden1_size),
            nn.LeakyReLU(),
            nn.Linear(hidden1_size, out_size)
        )

    def forward(self, o):
        return self.fc(o)
