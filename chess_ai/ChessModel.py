import torch
import torch.nn as nn
import torch.nn.functional as F

# loosely based on https://github.com/kuangliu/pytorch-cifar


class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ChessModel(nn.Module):
    def __init__(
        self, num_blocks=18, hidden_channels=128, in_channels=20, out_channels=73
    ):
        super().__init__()
        self.in_block = BasicBlock(in_channels, hidden_channels)
        self.res_blocks = nn.Sequential(
            *[BasicBlock(hidden_channels, hidden_channels) for _ in range(num_blocks)]
        )
        self.out_block = nn.Conv2d(
            hidden_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        res_in = self.in_block(x)
        res_out = self.res_blocks(res_in)
        return self.out_block(res_out)
