import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock(nn.Module):
    def __init__(self, in_channels=1, out_1x1=8, red_3x3=4, out_3x3=8, red_5x5=4, out_5x5=8, out_pool=8):
        super(InceptionBlock, self).__init__()

        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, red_3x3, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_3x3, out_3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, red_5x5, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(red_5x5, out_5x5, kernel_size=5, padding=2),
            nn.ReLU(inplace=True)
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        branch1x1_output = self.branch1x1(x)
        branch3x3_output = self.branch3x3(x)
        branch5x5_output = self.branch5x5(x)
        branch_pool_output = self.branch_pool(x)

        outputs = [branch1x1_output, branch3x3_output, branch5x5_output, branch_pool_output]
        return torch.cat(outputs, 1)  
    

    
class BasicConv(nn.Module):

    def   __init__(self, in_channels, out_channels, kernel=3, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.conv = nn.Sequential(
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel, padding="same"),
            nn.InstanceNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel, padding="same"),
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2),
            BasicConv(in_channels, out_channels, kernel)
        )

    def forward(self, x):
        x = self.down_conv(x)
        return x

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = BasicConv(in_channels, out_channels, kernel=kernel)
        
    def forward(self, x1, x2):
        y = self.up(x1)
        y  = torch.cat([x2, y], dim=1)
        y  = self.conv(y)
        return y

class Up_Only(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=3):
        super().__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = BasicConv(in_channels, out_channels, kernel=kernel)
        
    def forward(self, x1):
        y = self.up(x1)
        y  = self.conv(y)
        return y
    

class Outc(nn.Module):

    def __init__(self, in_channels, out_channels,mid_channels, kernel=3):
        super().__init__()
        self.conv = nn.Sequential(
            BasicConv(in_channels, mid_channels, kernel),
            BasicConv(mid_channels, out_channels, kernel),
            nn.Conv2d(out_channels, out_channels, kernel_size=1) 
        )
 
    def forward(self, x):
        x=self.conv(x)
        return x


# -------------------------
# Attention Gate
# -------------------------
class AttentionGate(nn.Module):
    def __init__(self, in_ch_x: int, in_ch_g: int, inter_ch: int):
        super().__init__()
        self.theta_x = nn.Conv2d(in_ch_x, inter_ch, kernel_size=1, bias=True)
        self.phi_g = nn.Conv2d(in_ch_g, inter_ch, kernel_size=1, bias=True)
        self.psi = nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, g):
        g_up = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)
        theta_x = self.theta_x(x)
        phi_g = self.phi_g(g_up)
        f = self.relu(theta_x + phi_g)
        psi = self.psi(f)
        alpha = self.sigmoid(psi)
        return x * alpha


# -------------------------
# Up + Attention + BasicConv
# -------------------------
class UpAttention(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        inter_ch = max(out_ch // 2, 1)
        self.att = AttentionGate(in_ch_x=skip_ch, in_ch_g=out_ch, inter_ch=inter_ch)
        self.conv = BasicConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        if diffX != 0 or diffY != 0:
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
        skip_att = self.att(skip, x)
        x = torch.cat([skip_att, x], dim=1)
        return self.conv(x)