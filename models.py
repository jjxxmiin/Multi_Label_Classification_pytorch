import os
import torch
import torch.nn as nn


def load_model(load_path, train, device='cuda'):
    model = VGG().to(device)

    if os.path.exists(load_path):
        model.load_state_dict(torch.load(load_path))

        if not train:
            model = model.eval()
        print("LOAD MODEL")
    else:
        print("NEW MODEL")

    return model

class VGG(nn.Module):
    def __init__(self, classes=20):
        super(VGG, self).__init__()
        """
        VGG 16
        layers = [2,2,3,3,3]
        chennels = [64,128,256,512,512]
        """
        self.vgg16 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classify = nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024),
            nn.Dropout(0.1),
            nn.Linear(1024, classes)
        )

    def forward(self, x):
        x = self.vgg16(x)
        x = x.view(x.size(0), -1)
        out = self.classify(x)

        return out

def test():
    net = VGG()
    x = torch.randn(2, 3, 224, 224)
    y = net(x)
    print(y.size())

#test()
