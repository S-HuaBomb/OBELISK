import torch
import torch.nn as nn
import torch.nn.functional as F


# Hybrid OBELISK model trained using pytorch v0.4.1 with TCIA data
# model contains two obelisk layers combined with traditional CNNs
# the layers have 512 and 128 trainable offsets and 230k trainable weights in total
class Obelisk_Unet(nn.Module):
    def __init__(self, num_labels, full_res):
        super(Obelisk_Unet, self).__init__()
        self.num_labels = num_labels
        D_in2 = full_res[0]  # 192
        H_in2 = full_res[1]  # 160
        W_in2 = full_res[2]  # 192
        D_in3 = D_in2 // 2  # 96
        H_in3 = H_in2 // 2  # 80
        W_in3 = W_in2 // 2  # 96

        D_in4 = D_in3 // 2  # 48
        H_in4 = H_in3 // 2  # 40
        W_in4 = W_in3 // 2  # 48

        D_in5 = D_in4 // 2  # 24
        H_in5 = H_in4 // 2  # 20
        W_in5 = W_in4 // 2  # 24

        D_grid1 = D_in3
        H_grid1 = H_in3
        W_grid1 = W_in3
        D_grid2 = D_in4
        H_grid2 = H_in4
        W_grid2 = W_in4
        D_grid3 = D_in5
        H_grid3 = H_in4
        W_grid3 = W_in5

        self.D_grid1 = D_grid1
        self.H_grid1 = H_grid1
        self.W_grid1 = W_grid1
        self.D_grid2 = D_grid2
        self.H_grid2 = H_grid2
        self.W_grid2 = W_grid2
        self.D_grid3 = D_grid3
        self.H_grid3 = H_grid3
        self.W_grid3 = W_grid3
        # U-Net Encoder
        self.conv0 = nn.Conv3d(1, 4, 3, padding=1)
        self.batch0 = nn.BatchNorm3d(4)
        self.conv1 = nn.Conv3d(4, 16, 3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm3d(16)
        self.conv11 = nn.Conv3d(16, 16, 3, padding=1)
        self.batch11 = nn.BatchNorm3d(16)
        self.conv2 = nn.Conv3d(16, 32, 3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch22 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, 32, 3, padding=1)
        self.batch3 = nn.BatchNorm3d(32)

        # Obelisk Encoder
        # sample_grid: 1 x    1     x #samples x 1 x 3
        # offsets:     1 x #offsets x     1    x 1 x 3
        self.sample_grid1 = torch.cat((torch.linspace(-1, 1, W_grid2).view(1, 1, -1, 1).repeat(D_grid2, H_grid2, 1, 1),
                                       torch.linspace(-1, 1, H_grid2).view(1, -1, 1, 1).repeat(D_grid2, 1, W_grid2, 1),
                                       torch.linspace(-1, 1, D_grid2).view(-1, 1, 1, 1).repeat(1, H_grid2, W_grid2, 1)),
                                      dim=3).view(1, 1, -1, 1, 3).detach()
        self.sample_grid1.requires_grad = False
        self.sample_grid2 = torch.cat((torch.linspace(-1, 1, W_grid3).view(1, 1, -1, 1).repeat(D_grid3, H_grid3, 1, 1),
                                       torch.linspace(-1, 1, H_grid3).view(1, -1, 1, 1).repeat(D_grid3, 1, W_grid3, 1),
                                       torch.linspace(-1, 1, D_grid3).view(-1, 1, 1, 1).repeat(1, H_grid3, W_grid3, 1)),
                                      dim=3).view(1, 1, -1, 1, 3).detach()
        self.sample_grid2.requires_grad = False
        self.offset1 = nn.Parameter(torch.randn(1, 128, 1, 1, 3) * 0.05)
        self.linear1a = nn.Conv3d(128, 128, 1, groups=2, bias=False)
        self.batch1a = nn.BatchNorm3d(128)
        self.linear1b = nn.Conv3d(128, 32, 1, bias=False)
        self.batch1b = nn.BatchNorm3d(128 + 32)
        self.linear1c = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.batch1c = nn.BatchNorm3d(128 + 64)
        self.linear1d = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.batch1d = nn.BatchNorm3d(128 + 96)
        self.linear1e = nn.Conv3d(128 + 96, 16, 1, bias=False)

        self.offset2 = nn.Parameter(torch.randn(1, 512, 1, 1, 3) * 0.05)
        self.linear2a = nn.Conv3d(512, 128, 1, groups=4, bias=False)
        self.batch2a = nn.BatchNorm3d(128)
        self.linear2b = nn.Conv3d(128, 32, 1, bias=False)
        self.batch2b = nn.BatchNorm3d(128 + 32)
        self.linear2c = nn.Conv3d(128 + 32, 32, 1, bias=False)
        self.batch2c = nn.BatchNorm3d(128 + 64)
        self.linear2d = nn.Conv3d(128 + 64, 32, 1, bias=False)
        self.batch2d = nn.BatchNorm3d(128 + 96)
        self.linear2e = nn.Conv3d(128 + 96, 32, 1, bias=False)

        # U-Net Decoder
        self.conv6bU = nn.Conv3d(64, 32, 3, padding=1)  # 96#64#32
        self.batch6bU = nn.BatchNorm3d(32)
        self.conv6U = nn.Conv3d(64, 12, 3, padding=1)  # 64#48
        self.batch6U = nn.BatchNorm3d(12)
        self.conv7U = nn.Conv3d(16, num_labels, 3, padding=1)  # 24#16#24
        self.batch7U = nn.BatchNorm3d(num_labels)
        self.conv77U = nn.Conv3d(num_labels, num_labels, 1)

    def forward(self, inputImg):
        B, C, D, H, W = inputImg.size()
        device = inputImg.device

        leakage = 0.025
        x0 = F.avg_pool3d(inputImg, 3, padding=1, stride=1)
        x00 = F.avg_pool3d(F.avg_pool3d(inputImg, 3, padding=1, stride=1), 3, padding=1, stride=1)

        x1 = F.leaky_relu(self.batch0(self.conv0(inputImg)), leakage)
        x = F.leaky_relu(self.batch1(self.conv1(x1)), leakage)
        x2 = F.leaky_relu(self.batch11(self.conv11(x)), leakage)
        x = F.leaky_relu(self.batch2(self.conv2(x2)), leakage)
        x = F.leaky_relu(self.batch22(self.conv22(x)), leakage)

        x = F.leaky_relu(self.batch3(self.conv3(x)), leakage)

        x_o1 = F.grid_sample(x0, (self.sample_grid1.to(device).repeat(B, 1, 1, 1, 1) + self.offset1)) \
            .view(B, -1, self.D_grid2, self.H_grid2, self.W_grid2)

        x_o1 = F.leaky_relu(self.linear1a(x_o1), leakage)
        x_o1a = torch.cat((x_o1, F.leaky_relu(self.linear1b(self.batch1a(x_o1)), leakage)), dim=1)
        x_o1b = torch.cat((x_o1a, F.leaky_relu(self.linear1c(self.batch1b(x_o1a)), leakage)), dim=1)
        x_o1c = torch.cat((x_o1b, F.leaky_relu(self.linear1d(self.batch1c(x_o1b)), leakage)), dim=1)
        x_o1d = F.leaky_relu(self.linear1e(self.batch1d(x_o1c)), leakage)
        x_o1 = F.interpolate(x_o1d, size=[self.D_grid1, self.H_grid1, self.W_grid1], mode='trilinear',
                             align_corners=False)

        x_o2 = F.grid_sample(x00, (self.sample_grid2.to(device).repeat(B, 1, 1, 1, 1) + self.offset2)) \
            .view(B, -1, self.D_grid3, self.H_grid3, self.W_grid3)

        x_o2 = F.leaky_relu(self.linear2a(x_o2), leakage)
        x_o2a = torch.cat((x_o2, F.leaky_relu(self.linear2b(self.batch2a(x_o2)), leakage)), dim=1)
        x_o2b = torch.cat((x_o2a, F.leaky_relu(self.linear2c(self.batch2b(x_o2a)), leakage)), dim=1)
        x_o2c = torch.cat((x_o2b, F.leaky_relu(self.linear2d(self.batch2c(x_o2b)), leakage)), dim=1)
        x_o2d = F.leaky_relu(self.linear2e(self.batch2d(x_o2c)), leakage)
        x_o2 = F.interpolate(x_o2d, size=[self.D_grid2, self.H_grid2, self.W_grid2], mode='trilinear',
                             align_corners=False)

        x = F.leaky_relu(self.batch6bU(self.conv6bU(torch.cat((x, x_o2), 1))), leakage)
        x = F.interpolate(x, size=[self.D_grid1, self.H_grid1, self.W_grid1], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch6U(self.conv6U(torch.cat((x, x_o1, x2), 1))), leakage)
        x = F.interpolate(x, size=[D, H, W], mode='trilinear', align_corners=False)
        x = F.leaky_relu(self.batch7U(self.conv7U(torch.cat((x, x1), 1))), leakage)

        x = self.conv77U(x)

        return x
