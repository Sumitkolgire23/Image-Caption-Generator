import torch.nn as nn
import torchvision.models as models
import torch

class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train_cnn=False):
        super().__init__()
        self.train_cnn = train_cnn
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove the FC layer
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = self.train_cnn

        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
            features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features
