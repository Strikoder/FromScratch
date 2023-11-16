#without mirroring padding - not losing too much performance

import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.models as models

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

class UNET(nn.Module):
    #outchannels in paper 2 here are 1 for binary
    def __init__(self, in_channels=3,out_channels=1,features=[64,128,256,512]) :
        super().__init__()
        self.downs=nn.ModuleList()
        self.ups=nn.ModuleList()
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels,feature))
            in_channels=feature

        self.bottleneck = DoubleConv(features[-1],features[-1]*2)

        # up part (Here we will use Transpose conv for the upsampling)
        for feature in reversed(features):
            self.ups.append(
                #feature*2 for skip connection
                nn.ConvTranspose2d(feature*2, feature, kernel_size=3,stride=2)
            )
            self.ups.append(DoubleConv(feature*2,feature))

        self.final_conv=nn.Conv2d(features[0],out_channels,1)

    def forward(self,x):
        skip_connections = []
        for down in self.downs:
            x=down(x)
            skip_connections.append(x)
            x=self.pool(x)
        
        x=self.bottleneck(x)
        skip_connections=skip_connections[::-1]
        # up and double conv with a single step
        for idx in range(0,len(self.ups),2):
            x=self.ups[idx](x)
            skip_connection=skip_connections[idx//2]
            # to correct the shape before concat if the skip connection was a different shape (161x161 with 80x80 so our ouput would be 160x160) or we should use an input divisable by 16 :D
            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])
            concat_skip=torch.concat((skip_connection,x),dim=1)
            x=self.ups[idx+1](concat_skip)
        
        return self.final_conv(x) #nn.sigmoid for binary
    
# def test():
#     x=torch.rand((3,1,160,160))
#     model=UNET(in_channels=1,out_channels=1)
#     preds=model(x)
#     print(preds.shape)
#     print(x.shape)
#     assert preds.shape==x.shape


