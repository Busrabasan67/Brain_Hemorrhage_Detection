import torch
import torch.nn as nn

class PureCustomCNN_v6(nn.Module):
    def __init__(self):
        super(PureCustomCNN_v6, self).__init__()

        def double_conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        def single_conv_block_with_spatial_dropout(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(0.1)
            )

        self.layer1 = double_conv_block(3, 32)
        self.layer2 = single_conv_block_with_spatial_dropout(32, 64)
        self.layer3 = single_conv_block_with_spatial_dropout(64, 128)
        self.layer4 = single_conv_block_with_spatial_dropout(128, 256)

        self.flatten = nn.Flatten()

        self.fc = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(256 * 14 * 14, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.flatten(x)
        return self.fc(x)

def get_custom_cnn_v6():
    return PureCustomCNN_v6()
