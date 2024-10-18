import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh):
        super(CRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.cnn = nn.Sequential(
            nn.Conv2d(nc, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2, (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0),
            nn.ReLU(True)
        )

        self.rnn = nn.Sequential(
            nn.LSTM(512, nh, bidirectional=True, num_layers=2),
            nn.Linear(nh * 2, nclass)
        )

    def forward(self, input):
        # Pass through CNN
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "Height of feature map must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)

        # Pass through RNN
        output, _ = self.rnn(conv)
        return output
