"""A VGG-based perceptual loss function for PyTorch."""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models, transforms

class VGGLoss(nn.Module):
    """Computes the VGG perceptual loss between two batches of images.

    The input and target must be 4D tensors with three channels
    ``(B, 3, H, W)`` and must have equivalent shapes. Pixel values should be
    normalized to the range 0–1.

    The VGG perceptual loss is the mean squared difference between the features
    computed for the input and target at layer :attr:`layer` (default 8, or
    ``relu2_2``) of the pretrained model specified by :attr:`model` (either
    ``'vgg16'`` (default) or ``'vgg19'``).

    If :attr:`shift` is nonzero, a random shift of at most :attr:`shift`
    pixels in both height and width will be applied to all images in the input
    and target. The shift will only be applied when the loss function is in
    training mode, and will not be applied if a precomputed feature map is
    supplied as the target.

    :attr:`reduction` can be set to ``'mean'``, ``'sum'``, or ``'none'``
    similarly to the loss functions in :mod:`torch.nn`. The default is
    ``'mean'``.

    :meth:`get_features()` may be used to precompute the features for the
    target, to speed up the case where inputs are compared against the same
    target over and over. To use the precomputed features, pass them in as
    :attr:`target` and set :attr:`target_is_features` to :code:`True`.

    Instances of :class:`VGGLoss` must be manually converted to the same
    device and dtype as their inputs.
    """

    models = {'vgg16': models.vgg16_bn, 'vgg19': models.vgg19_bn}

    def __init__(self, model='vgg19', layer=8, shift=0, reduction='mean'):
        super().__init__()
        self.shift = shift
        self.reduction = reduction
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.model = self.models[model](pretrained=True).features
        self.model.eval()
        self.model.requires_grad_(False)


    def get_features(self, input):
        return self.model(self.normalize(input))

    def get_multi_features(self, input):
        input = input.permute(0, 3, 1, 2)
        # output.append(input)
        x = self.normalize(input)
        required_layers = [17]
        for index,layer in enumerate(self.model):
            x = layer(x)
            if index in required_layers:
                output= x
                break
        return output

    def train(self, mode=True):
        self.training = mode


    def forward(self, input, target, target_is_features=False):
        if target_is_features:
            input_feats = self.get_features(input)
            target_feats = target
        else:
            sep = input.shape[0]
            batch = torch.cat([input, target])
            if self.shift and self.training:
                padded = F.pad(batch, [self.shift] * 4, mode='replicate')
                batch = transforms.RandomCrop(batch.shape[2:])(padded)
            feats = self.get_features(batch)
            input_feats, target_feats = feats[:sep], feats[sep:]
        return F.mse_loss(input_feats, target_feats, reduction=self.reduction)


if __name__=='__main__':
    input = torch.randn([1,3,400,400])
    target = torch.randn([1,3,400,400])
    model_vgg = VGGLoss()
    loss = model_vgg(input, target)