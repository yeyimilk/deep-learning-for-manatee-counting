
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from nn_utils import Conv2d

class VGG(nn.Module):
    def __init__(self, pretrained=True):
        super(VGG, self).__init__()
        self.seen = 0
        self.de_pred = nn.Sequential(Conv2d(512, 128, 1, same_padding=True, NL='relu'),
                                     Conv2d(128, 1, 1, same_padding=True, NL='relu'))
        
        self._initialize_weights()
        
        vgg = models.vgg16(pretrained=pretrained)
        # if pretrained:
        #     vgg.load_state_dict(torch.load(model_path))
        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])

    def forward(self, x):
        x = self.features4(x)       
        x = self.de_pred(x)

        x = F.upsample(x,scale_factor=8)

        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)