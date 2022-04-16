import torch
import torch.nn as nn

# double convolutions
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, layer):
        return self.double_conv(layer)

# down sampling
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, layer):
        return self.maxpool_conv(layer)

# up sampling
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear = True):
        super().__init__()

        if bilinear:
            # normal convolutions to reduce the number of channels
            self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, (in_channels // 2), kernel_size = 2, stride = 2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, layer1, layer2):
        layer1 = self.up(layer1)
        diffY = torch.tensor([layer2.size()[2] - layer1.size()[2]])
        diffX = torch.tensor([layer2.size()[3] - layer1.size()[3]])

        layer1 = nn.functional.pad(layer1, [(diffX // 2), (diffX - (diffX // 2)), (diffY // 2), (diffY - (diffY // 2))])
        layer = torch.cat([layer2, layer1], dim = 1)
        return self.conv(layer)

# output convolution layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1)

    def forward(self, layer):
        return self.conv(layer)

# U-Net architecture
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = False):
        super(UNet, self).__init__()
        self.input = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512, bilinear)
        self.up2 = Up(512, 256, bilinear)
        self.up3 = Up(256, 128, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.output = OutConv(64, n_classes)

    def forward(self, layer):
        layer_init = self.input(layer)
        down_layer_1 = self.down1(layer_init)
        down_layer_2 = self.down2(down_layer_1)
        down_layer_3 = self.down3(down_layer_2)
        down_layer_4 = self.down4(down_layer_3)
        up_layer_1 = self.up1(down_layer_4, down_layer_3)
        up_layer_2 = self.up2(up_layer_1, down_layer_2)
        up_layer_3 = self.up3(up_layer_2, down_layer_1)
        up_layer_4 = self.up4(up_layer_3, layer_init)
        logits = self.output(up_layer_4)
        return logits

if '__main__' == __name__:
    net = UNet(n_channels = 3, n_classes = 1)
    print(net)
