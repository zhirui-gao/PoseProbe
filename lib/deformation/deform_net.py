# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Define DIF-Net
'''
import torch.nn.init as init
import torch
from torch import nn
from lib.deformation import modules
import torch.nn.functional as F
from lib.embedder import get_embedder
class DeformedImplicitField(nn.Module):
    def __init__(self,  model_type='relu', hyper_hidden_layers=1,range_shape=None,
                 hyper_hidden_features=256, hidden_num=128, **kwargs):
        super().__init__()
        self.progress = torch.nn.Parameter(torch.tensor(0.))
        self.output_range = range_shape.max()
        # Deform-Net
        self.deform_net = modules.SingleBVPNet(type=model_type, mode='mlp', hidden_features=hidden_num,
                                               num_hidden_layers=3, in_features=3, out_features=4)


    def forward(self, model_input):
        model_output = [self.deform_net(x)['model_out'] for x in model_input.split(8192*2, 0)] #
        model_output = torch.cat(model_output, dim=0)
        model_output = model_output * self.output_range
        deformation = model_output[:, :3]  # 3 dimensional deformation field
        correction = model_output[:, 3:]  # scalar correction field
        return deformation , correction








