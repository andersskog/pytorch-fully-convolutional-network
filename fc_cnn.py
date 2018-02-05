import torch
import torch.nn as nn
import torchvision
from torch.optim import lr_scheduler

from CONSTANTS import NUM_CLASSES, USE_CUDA

class VGG_FCN_8(nn.Module):

    def __init__(self):
        super(VGG_FCN_8, self).__init__()
        self.features = nn.Sequential(*list(torchvision.models.vgg16(pretrained=True).features))
        self.upsampler_x32_sequence = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7),
                                                    nn.ReLU(True),
                                                    nn.Dropout(),
                                                    nn.Conv2d(4096, 4096, kernel_size=1),
                                                    nn.ReLU(True),
                                                    nn.Dropout(),
                                                    nn.Conv2d(4096, NUM_CLASSES, kernel_size=1)
                                                   )
        self.upsampler_x32 = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, kernel_size=4, stride=2, bias=False)
        self.upsampler_x16_sequence = nn.Sequential(nn.Conv2d(512, NUM_CLASSES, kernel_size=1))
        self.upsampler_x16 = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES,  kernel_size=4, stride=2, bias=False)                                    
        self.upsampler_x8_sequence = nn.Sequential(nn.Conv2d(256, NUM_CLASSES, kernel_size=1))
        self.upsampler_x8 = nn.ConvTranspose2d(NUM_CLASSES, NUM_CLASSES, kernel_size=16, stride=8, bias=False)
        
        self.features[0].padding = (100, 100)
        
    def forward(self, x):
        x_size = x.size()
        output = x
        for i in range(17):
            output = self.features[i](output)
        upsample_x8 = self.upsampler_x8_sequence(0.0001 * output)
        for i in range(17, 24):
            output = self.features[i](output)
        upsample_x16 = self.upsampler_x16_sequence(0.01 * output)
        for i in range(24, 31):
            output = self.features[i](output)
        upsample_x32 = self.upsampler_x32_sequence(output)
        upscore_x32 = self.upsampler_x32(upsample_x32)
        upscore_x16 = self.upsampler_x16(upsample_x16[:, :, 5: (5 + upscore_x32.size()[2]), 5: (5 + upscore_x32.size()[3])] + upscore_x32) 
        upscore_x8 = self.upsampler_x8(upsample_x8[:, :, 9: (9 + upscore_x16.size()[2]), 9: (9 + upscore_x16.size()[3])] + upscore_x16)
        upscore_x8 = upscore_x8[:, :, 31: (31 + x_size[2]), 31: (31 + x_size[3])].contiguous()
        return upscore_x8

def initialize_model():
    model_conv = VGG_FCN_8()
    # freeze vgg params
    for param in model_conv.features.parameters():
        param.requires_grad = False
    # Parameters of newly constructed modules have requires_grad=True by default
    if USE_CUDA:
        model_conv = model_conv.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer_conv = torch.optim.SGD(filter(lambda p: p.requires_grad,  model_conv.parameters()), lr=1e-3, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
    return model_conv, optimizer_conv, criterion