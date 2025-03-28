"""A VGG-based perceptual loss function for PyTorch."""

import torch
import torchvision.models.detection
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms
# from lib.resnet import resnet101, ResNet101_Weights
class ResNetFpn_Loss(nn.Module):

    def __init__(self, model='vgg16', layer=8, shift=0, reduction='mean'):
        super().__init__()

        self.model =  torchvision.models.resnet101(pretrained=True) #resnet101(weights = ResNet101_Weights)
        self.model.eval()
        self.model.requires_grad_(False)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    def get_features(self, input):
        return self.model(self.normalize(input))

    def get_multi_features(self, input):
        output = []
        input = input.permute(0,3,1,2)
        output.append(input)
        x = self.normalize(input)
        required_layers = [2, 4, 5]
        for index, module in enumerate(self.model.named_children()):
            name, model = module
            x = model(x)
            if index in required_layers:
                output.append(x)
            if index>= required_layers[-1]:
                break

        return output




if __name__=='__main__':
    input = torch.randn([1,3,400,400])
    target = torch.randn([1,3,400,400])
    model_vgg = ResNetFpn_Loss()
    loss = model_vgg(input, target)