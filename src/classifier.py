import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):
    def __init__(self, num_classes, img_size=40):
        super(Classifier, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.padding = "same"

        self.conv1 = self.cnn_layer(3, 32)  # (32, 40, 40)
        self.conv2 = self.cnn_layer(32, 32)  # (32, 40, 40)
        self.pool1 = nn.MaxPool2d(2)  # (32, 20, 20)

        self.conv3 = self.cnn_layer(32, 64)  # (64, 20, 20)
        self.conv4 = self.cnn_layer(64, 64)  # (64, 20, 20)
        self.conv5 = self.cnn_layer(64, 64)  # (64, 20, 20)
        self.conv6 = self.cnn_layer(64, 64)  # (64, 20, 20)
        self.pool2 = nn.MaxPool2d(2)  # (64, 10, 10)

        self.conv7 = self.cnn_layer(64, 128)  # (128, 10, 10)
        self.pool3 = nn.AvgPool2d(10)  # (128, 1, 1)

        self.fc1 = nn.Linear(128, 256)  # (256)

        self.head = nn.Linear(256, self.num_classes)  # (num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool2(x)

        x = self.conv7(x)
        x = self.pool3(x)

        x = x.view(-1, 128)
        x = self.fc1(x)

        x = self.head(x)
        x = F.softmax(x, dim=1)

        return x

    def cnn_layer(self, in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=self.padding),
            nn.BatchNorm2d(out_channels),
        )

