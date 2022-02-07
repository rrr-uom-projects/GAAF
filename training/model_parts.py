import torch.nn as nn
import torch.nn.functional as F

class conv_module(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop=0.25):
        super(conv_module, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(p=p_drop),
            nn.Conv3d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(p=p_drop),
        )
        if in_channels != out_channels:
            self.res_conv = nn.Sequential(
                nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1,1,1)),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            )
        else:
            self.res_conv = None

    def forward(self, x):
        if self.res_conv is not None:
            return self.double_conv(x) + self.res_conv(x)
        else:
            return self.double_conv(x) + x


class resize_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2)):
        super(resize_conv, self).__init__()
        self.resize_conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3,3), padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(p=p_drop),
        )
        self.scale_factor = scale_factor

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='trilinear', align_corners=False)
        return self.resize_conv(x)


class transpose_conv(nn.Module):
    def __init__(self, in_channels, out_channels, p_drop, scale_factor=(2,2,2)):
        super(transpose_conv, self).__init__()
        self.transpose_conv = nn.Sequential(
            nn.ConvTranspose3d(in_channels=in_channels, out_channels=out_channels, kernel_size=scale_factor, stride=scale_factor), # stride & kernel (1,2,2) gives (D_in, 2*H_in, 2*W_in)
            nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Dropout3d(p=p_drop),
        )

    def forward(self, x):
        return self.transpose_conv(x)


class attention_module(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(attention_module,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1,1,1)),
            nn.BatchNorm3d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1,1,1)),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=(1,1,1)),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = F.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi