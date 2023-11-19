"""
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision.models import vgg16_bn
from einops.layers.torch import Rearrange

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv=nn.Sequential(
            #no bias since its gonna be cancled by batchnorm
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            #was not in the paper (released before it)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x) 

class UNETR(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super(UNETR, self).__init__()

        # VGG Backbone
        self.vgg_backbone = vgg16_bn(pretrained=True).features[:33]  # Using first 33 layers of VGG16 with batch normalization
        self.in_channels = in_channels

        # Transformer
        self.transformer = nn.Transformer()  # You can customize this part based on your specific requirements
        
        # Convolution layers
        self.conv_layers = nn.Sequential(
            DoubleConv(features[-1], features[-1] * 2),
            nn.ConvTranspose2d(features[-1] * 2, features[-1], kernel_size=2, stride=2),
            DoubleConv(features[-1] * 2, features[-1]),
            nn.Conv2d(features[-1], out_channels, kernel_size=1)
        )

    def forward(self, x):
        # Process through VGG Backbone
        x = self.vgg_backbone(x)
        
        x = torch.nn.functional.adaptive_avg_pool2d(x, (24, 24))

        # Rearrange for transformer and process
        
        x = Rearrange('b c h w -> b (h w) c')(x)
        x = self.transformer(x, x)
        x = Rearrange('b (h w) c -> b c h w', h=int(x.shape[1]**0.5), w=int(x.shape[1]**0.5))(x)

        self.conv1 = nn.Conv2d(512, 1024, kernel_size=3, padding=1, bias=False)  # Adjusting to 512 input channels 
        print(f'x before conv1{x.shape}')
        x = self.conv1(x)
        print(f'x after conv1{x.shape}')

        x = self.conv1(x)
        # Process through convolution layers
        x = self.conv_layers(x)
        return x
"""


import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv=nn.Sequential(
            #no bias since its gonna be cancled by batchnorm
            nn.Conv2d(in_channels,out_channels,3,1,1,bias=False),
            #was not in the paper (released before it)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels,out_channels,3,1,1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.conv(x) 

class UNETWithBackbone(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]):
        super().__init__()

        # Load pre-trained VGG16 model
        self.vgg_backbone = models.vgg16(pretrained=True).features[:23]  # Extract features from VGG16 up to the 23rd layer

        self.bottleneck = DoubleConv(512, 1024)

        # Upsampling layers
        self.ups = nn.ModuleList()
        self.ups.append(nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2))  # From bottleneck
        self.ups.append(DoubleConv(512 + 256, 512))  # Concatenating with skip connection (256 channels)
        self.ups.append(nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2))  # Reduce 512 to 256
        self.ups.append(DoubleConv(256 + 128, 256))  # Concatenating with skip connection (128 channels)
        self.ups.append(nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2))  # Reduce 256 to 128
        self.ups.append(DoubleConv(128 + 64, 128))   # Concatenating with skip connection (64 channels)
        self.ups.append(nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2))  # Reduce 128 to 64
        self.ups.append(DoubleConv(64, 64))          # Final DoubleConv layer




        # Final convolutional layer
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)


    def forward(self, x):

        skip_connections = []

        # Passing input through the VGG16 backbone and collecting skip connections
        for idx, layer in enumerate(self.vgg_backbone):
            x = layer(x)
            if idx in {4, 9, 16, 27}:  # Adjust these indices based on your needs
                skip_connections.append(x)

        x = self.bottleneck(x)
        # Reversing skip connections
        skip_connections = skip_connections[::-1]

        # Iterate over the upsampling layers
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)

            if len(skip_connections) > 0:
                skip_connection = skip_connections.pop()
                # Resize if shapes are different
                if x.shape != skip_connection.shape:
                    x = TF.resize(x, size=skip_connection.shape[2:])
                x = torch.cat((x, skip_connection), dim=1)

        x = self.ups[idx + 1](x)

        return self.final_conv(x)
