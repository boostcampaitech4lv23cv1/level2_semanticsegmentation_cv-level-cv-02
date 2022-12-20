import torch.nn as nn
import torch.nn.functional as F
import torchvision


class FCN_ResNet50(nn.Module):
    def __init__(self, num_classes = 11):
        super(FCN_ResNet50, self).__init__()
        self.model = torchvision.models.segmentation.fcn_resnet50(pretrained=True)
        self.model.classifier[4] = nn.Conv2d(512, 11, kernel_size=1)
    
    def forward(self, x):
        x = self.model(x)
        return x