import torch
from torch import nn


class ConvGRUCell2D(nn.Module):
    def __init__(self, in_channels):
        """Initialize the ConvLSTM cell"""
        super().__init__()
        self.in_channels = in_channels

        self.conv_gates_1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_gates_2 = nn.Sequential(
            BasicConvolutionBlock2D(
                in_channels, in_channels, ks=2, stride=2, dilation=1
            ),
            ResidualBlock2D(
                in_channels, in_channels, ks=3, stride=1, dilation=1, padding=1
            ),
            BasicDeconvolutionBlock2D(in_channels, in_channels, ks=2, stride=2),
        )

        self.conv_gates_3 = nn.Conv2d(
            in_channels, in_channels * 2, kernel_size=3, stride=1, padding=1
        )

        self.conv_can_1 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
        )

        self.conv_can_2 = nn.Sequential(
            BasicConvolutionBlock2D(
                in_channels, in_channels, ks=2, stride=2, dilation=1
            ),
            ResidualBlock2D(
                in_channels, in_channels, ks=3, stride=1, dilation=1, padding=1
            ),
            BasicDeconvolutionBlock2D(in_channels, in_channels, ks=2, stride=2),
        )

        self.conv_can_3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, cur, mem):
        # cur.shape == mem.shape == [B, C, H, W]
        combined = torch.cat([cur, mem], dim=1)

        combined_feat_1 = self.conv_gates_1(combined)
        combined_feat_2 = self.conv_gates_2(combined_feat_1)
        combined_feat_2 = self.conv_gates_3(combined_feat_1 + combined_feat_2)

        gamma, beta = torch.split(combined_feat_2, self.in_channels, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([cur, reset_gate * mem], dim=1)

        combined_feat_1 = self.conv_can_1(combined)
        combined_feat_2 = self.conv_can_2(combined_feat_1)
        combined_feat_2 = self.conv_can_3(combined_feat_1 + combined_feat_2)

        cnm = torch.tanh(combined_feat_2)
        h_next = (1 - update_gate) * mem + update_gate * cnm

        return h_next


class BasicConvolutionBlock2D(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, dilation=1, norm_getter=None):
        super().__init__()

        if norm_getter is None:
            norm_getter = nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=ks, dilation=dilation, stride=stride),
            norm_getter(outc),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        out = self.net(x)
        return out


class BasicDeconvolutionBlock2D(nn.Module):
    def __init__(self, inc, outc, ks=3, stride=1, norm_getter=None):
        super().__init__()

        if norm_getter is None:
            norm_getter = nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.ConvTranspose2d(inc, outc, kernel_size=ks, stride=stride),
            norm_getter(outc),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ResidualBlock2D(nn.Module):
    def __init__(
        self,
        inc,
        outc,
        ks=3,
        stride=1,
        dilation=1,
        inplace_relu=True,
        norm_getter=None,
        padding=1,
    ):
        super().__init__()

        if norm_getter is None:
            norm_getter = nn.BatchNorm2d

        self.net = nn.Sequential(
            nn.Conv2d(
                inc,
                outc,
                kernel_size=ks,
                dilation=dilation,
                stride=stride,
                padding=padding,
            ),
            norm_getter(outc),
            nn.LeakyReLU(inplace=inplace_relu),
            nn.Conv2d(
                outc, outc, kernel_size=ks, dilation=dilation, stride=1, padding=padding
            ),
            norm_getter(outc),
        )

        if inc == outc and stride == 1:
            self.downsample = nn.Sequential()
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    inc, outc, kernel_size=1, dilation=1, stride=stride, padding=padding
                ),
                nn.BatchNorm2d(outc),
            )

        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.net(x) + self.downsample(x))
        return out
