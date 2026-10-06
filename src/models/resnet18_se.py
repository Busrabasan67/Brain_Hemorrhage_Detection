import torch
import torch.nn as nn
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

def get_resnet18_se(num_classes=1):
    # Load pretrained ResNet18
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Add SEBlock after layer4
    in_features = model.fc.in_features  # should be 512 for resnet18
    se_block = SEBlock(channel=in_features, reduction=16)
    
    # We can wrap layer4 with SE block
    model.layer4 = nn.Sequential(
        model.layer4,
        se_block
    )
    
    # Replace final fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model
