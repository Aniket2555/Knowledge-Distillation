import torch.nn as nn
from torchvision.models import resnet18

class TeacherNet(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherNet, self).__init__()
        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)
