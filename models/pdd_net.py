import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class OBELISK(nn.Module):
    def __init__(self, full_res=None, corner=False):
        super(OBELISK, self).__init__()

        if full_res is None:
            full_res = [160, 192, 160]
        self.o_m = full_res[0] // 3  # H//3
        self.o_n = full_res[1] // 3  # W//3
        self.o_o = full_res[2] // 3  # D//3

        self.ogrid_xyz = F.affine_grid(torch.eye(3, 4).unsqueeze(0), [1, 1, self.o_m, self.o_n, self.o_o],
                                       align_corners=corner).view(1, 1, -1, 1, 3)
        channels = 24  # 16
        self.offsets = nn.Parameter(torch.randn(2, channels * 2, 3) * 0.05)
        self.layer0 = nn.Conv3d(1, 4, 5, stride=2, bias=False, padding=2)
        self.batch0 = nn.BatchNorm3d(4)

        self.layer1 = nn.Conv3d(channels * 8, channels * 4, 1, bias=False, groups=1)
        self.batch1 = nn.BatchNorm3d(channels * 4)
        self.layer2 = nn.Conv3d(channels * 4, channels * 4, 3, bias=False, padding=1)
        self.batch2 = nn.BatchNorm3d(channels * 4)
        self.layer3 = nn.Conv3d(channels * 4, channels * 1, 1)

    def forward(self, input_img):
        img_in = F.avg_pool3d(input_img, 3, padding=1, stride=2)
        img_in = F.relu(self.batch0(self.layer0(img_in)))
        sampled = F.grid_sample(img_in, self.ogrid_xyz + self.offsets[0, :, :].view(1, -1, 1, 1, 3),
                                align_corners=self.corner).view(1, -1, self.o_m, self.o_n, self.o_o)
        sampled -= F.grid_sample(img_in, self.ogrid_xyz + self.offsets[1, :, :].view(1, -1, 1, 1, 3),
                                 align_corners=self.corner).view(1, -1, self.o_m, self.o_n, self.o_o)

        x = F.relu(self.batch1(self.layer1(sampled)))
        x = F.relu(self.batch2(self.layer2(x)))
        features = self.layer3(x)
        return features


class subplanar_pdd(nn.Module):
    def __init__(self, grid_size=29, displacement_width=15, disp_range=0.4, corner=False):
        super(subplanar_pdd, self).__init__()
        self.grid_size = grid_size  # number of control points per dimension
        self.grid_xyz = F.affine_grid(torch.eye(3, 4).unsqueeze(0), [1, 1, grid_size, grid_size, grid_size],
                                      align_corners=corner).view(1, -1, 1, 1, 3)
        self.displacement_width = displacement_width  # number of steps per dimension
        shift_xyz = F.affine_grid(disp_range * torch.eye(3, 4).unsqueeze(0),
                                  [1, 1, displacement_width, displacement_width, displacement_width],
                                  align_corners=corner).view(1, 1, -1, 1, 3)
        shift_x = shift_xyz.view(displacement_width, displacement_width, displacement_width, 3)[
                  (displacement_width - 1) // 2, :, :, :].reshape(1, 1, -1, 1, 3)
        shift_y = shift_xyz.view(displacement_width, displacement_width, displacement_width, 3)[:,
                  (displacement_width - 1) // 2, :, :].reshape(1, 1, -1, 1, 3)
        shift_z = shift_xyz.view(displacement_width, displacement_width, displacement_width, 3)[:, :,
                  (displacement_width - 1) // 2, :].reshape(1, 1, -1, 1, 3)
        self.shift_2d = torch.cat((shift_x, shift_y, shift_z), 3)

        self.corner = corner
        self.alpha = nn.Parameter(torch.Tensor([1, .1, 1, 1, .1, 5]))

        self.pad1 = nn.ReplicationPad3d((0, 0, 2, 2, 2, 2))
        self.avg1 = nn.AvgPool3d((3, 3, 1), stride=1)
        self.max1 = nn.MaxPool3d((3, 3, 1), stride=1)
        self.pad2 = nn.ReplicationPad3d((0, 0, 2, 2, 2, 2))

    def get_attributes(self):
        shift_2d_min = self.shift_2d.repeat(1, self.grid_size ** 3, 1, 1, 1)
        return self.shift_2d, shift_2d_min, self.grid_xyz

    def forward(self, feat00, feat50, shift_2d_min):
        # pdd correlation layer with 2.5D decomposition (slightly unrolled)
        pdd_cost = torch.zeros(1, self.grid_size ** 3, self.displacement_width, self.displacement_width, 3)
        xyz8 = self.grid_size ** 2
        for i in range(self.grid_size):
            moving_unfold = F.grid_sample(feat50,
                                          self.grid_xyz[:, i * xyz8:(i + 1) * xyz8, :, :, :] +
                                          shift_2d_min[:, i * xyz8:(i + 1) * xyz8, :, :, :],
                                          padding_mode='border',
                                          align_corners=self.corner)
            fixed_grid = F.grid_sample(feat00,
                                       self.grid_xyz[:, i * xyz8:(i + 1) * xyz8, :, :, :],
                                       align_corners=self.corner)
            pdd_cost[:, i * xyz8:(i + 1) * xyz8, :, :, :] = \
                self.alpha[1] + self.alpha[0] * torch.sum(torch.pow(fixed_grid - moving_unfold, 2), 1) \
                    .view(1, -1, self.displacement_width, self.displacement_width, 3)

        pdd_cost = pdd_cost.view(1, -1, self.displacement_width, self.displacement_width, 3)

        # approximate min convolution / displacement compatibility
        cost = (self.avg1(-self.max1(-self.pad1(pdd_cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2, 3, 4, 0, 1) \
            .view(1, 3 * self.displacement_width ** 2, self.grid_size, self.grid_size, self.grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0, 2, 3, 4, 1) \
            .view(1, -1, self.displacement_width, self.displacement_width, 3)

        # second path
        cost = self.alpha[4] + self.alpha[2] * pdd_cost + self.alpha[3] * cost_avg
        cost = (self.avg1(-self.max1(-self.pad1(cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2, 3, 4, 0, 1) \
            .view(1, 3 * self.displacement_width ** 2, self.grid_size, self.grid_size, self.grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0, 2, 3, 4, 1) \
            .view(self.grid_size ** 3, self.displacement_width ** 2, 3)
        # probabilistic and continuous output
        cost_soft = F.softmax(-self.alpha[5] * cost_avg, 1) \
            .view(-1, 1, self.displacement_width, self.displacement_width, 3)
        pred_xyz = 0.5 * (cost_soft.view(-1, self.displacement_width ** 2, 3, 1) *
                          self.shift_2d.view(1, self.displacement_width ** 2, 3, 3)).sum(1).sum(1)

        return cost_soft, pred_xyz, cost_avg


class Deeds(nn.Module):
    def __init__(self, grid_size=29, displacement_width=15, disp_range=0.4, corner=False):
        super(Deeds, self).__init__()

        self.grid_size = grid_size
        self.displacement_width = displacement_width
        self.grid_xyz = F.affine_grid(torch.eye(3, 4).unsqueeze(0), [1, 1, grid_size, grid_size, grid_size],
                                      align_corners=corner).view(1, -1, 1, 1, 3)
        self.shift_xyz = F.affine_grid(disp_range * torch.eye(3, 4).unsqueeze(0),
                                       [1, 1, displacement_width, displacement_width, displacement_width],
                                       align_corners=corner).view(1, 1, -1, 1, 3)

        self.alpha = nn.Parameter(torch.Tensor([1, .1, 1, 1, .1, 1]))

        self.pad1 = nn.ReplicationPad3d(3)
        self.avg1 = nn.AvgPool3d(3, stride=1)
        self.max1 = nn.MaxPool3d(3, stride=1)
        self.pad2 = nn.ReplicationPad3d(2)

    def forward(self, feat00, feat50):
        # deeds correlation layer (slightly unrolled)
        deeds_cost = torch.zeros(1, self.grid_size ** 3, self.displacement_width, self.displacement_width,
                                 self.displacement_width)
        xyz8 = self.grid_size ** 2
        for i in range(self.grid_size):
            moving_unfold = F.grid_sample(feat50, self.grid_xyz[:, i * xyz8:(i + 1) * xyz8, :, :, :] + self.shift_xyz,
                                          padding_mode='border')
            fixed_grid = F.grid_sample(feat00, self.grid_xyz[:, i * xyz8:(i + 1) * xyz8, :, :, :])
            deeds_cost[:, i * xyz8:(i + 1) * xyz8, :, :, :] = self.alpha[1] + self.alpha[0] * torch.sum(
                torch.pow(fixed_grid - moving_unfold, 2), 1).view(1, -1, self.displacement_width,
                                                                  self.displacement_width,
                                                                  self.displacement_width)

        # remove mean (not really necessary)
        # deeds_cost = deeds_cost.view(-1,displacement_width**3) - deeds_cost.view(-1,displacement_width**3).mean(1,keepdim=True)[0]
        deeds_cost = deeds_cost.view(1, -1, self.displacement_width, self.displacement_width, self.displacement_width)

        # approximate min convolution / displacement compatibility
        cost = self.avg1(self.avg1(-self.max1(-self.pad1(deeds_cost))))

        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2, 3, 4, 0, 1)\
            .view(1, self.displacement_width ** 3, self.grid_size, self.grid_size, self.grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0, 2, 3, 4, 1) \
            .view(1, -1, self.displacement_width, self.displacement_width, self.displacement_width)

        # second path
        cost = self.alpha[4] + self.alpha[2] * deeds_cost + self.alpha[3] * cost_avg
        cost = self.avg1(self.avg1(-self.max1(-self.pad1(cost))))
        # grid-based mean field inference (one iteration)
        cost_permute = cost.permute(2, 3, 4, 0, 1)\
            .view(1, self.displacement_width ** 3, self.grid_size, self.grid_size, self.grid_size)
        cost_avg = self.avg1(self.avg1(self.pad2(cost_permute))).permute(0, 2, 3, 4, 1)\
            .view(self.grid_size ** 3, self.displacement_width ** 3)
        # cost = alpha[4]+alpha[2]*deeds_cost+alpha[3]*cost.view(1,-1,displacement_width,displacement_width,displacement_width)
        # cost = avg1(avg1(-max1(-pad1(cost))))

        # probabilistic and continuous output
        cost_soft = F.softmax(-self.alpha[5] * cost_avg, 1)
        #        pred_xyz = torch.sum(F.softmax(-5self.alpha[2]*cost_avg,1).unsqueeze(2)*shift_xyz.view(1,-1,3),1)
        pred_xyz = torch.sum(cost_soft.unsqueeze(2) * self.shift_xyz.view(1, -1, 3), 1)

        return cost_soft, pred_xyz


# GridNet and fit_sub2dense are used for instance optimisation (fitting of 2.5D displacement costs)
class GridNet(nn.Module):
    def __init__(self, grid_x, grid_y, grid_z):
        super(GridNet, self).__init__()
        self.params = nn.Parameter(0.1 * torch.randn(1, 3, grid_x, grid_y, grid_z))

    def forward(self):
        return self.params


def fit_sub2dense(pred_xyz, grid_xyz, cost_avg, alpha,
                  full_res=None, disp_range=0.4, displacement_width=15, lambda_w=1.5, max_iter=100):
    if full_res is None:
        full_res = [160, 192, 160]
    H, W, D = full_res[0], full_res[1], full_res[2]
    H2 = H // 3
    W2 = W // 3
    D2 = D // 3
    cost2d = F.softmax(-alpha[5] * cost_avg, 1).view(-1, 1, displacement_width, displacement_width, 3)

    with torch.enable_grad():
        net = GridNet(H2, W2, D2)
        net.params.data = pred_xyz.permute(0, 4, 1, 2, 3).detach() + torch.randn_like(
            pred_xyz.permute(0, 4, 1, 2, 3)) * 0.05
        net.cuda()
        avg5 = nn.AvgPool3d((3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)).cuda()

        optimizer = optim.Adam(net.parameters(), lr=0.02)
        lambda_weight = lambda_w  # 1.5#5

        for _ in range(max_iter):
            optimizer.zero_grad()
            # second-order B-spline transformation model
            fitted_grid = (avg5(avg5(net())))
            # resampling transformation network to chosen control point spacing
            sampled_net = F.grid_sample(fitted_grid, grid_xyz, align_corners=True).permute(2, 0, 3, 4,
                                                                                           1) / disp_range
            # sampling the 2.5D displacement probabilities at 3D vectors
            sampled_cost = 0.33 * F.grid_sample(cost2d[:, :, :, :, 0], sampled_net[:, :, :, 0, :2],
                                                align_corners=True)
            sampled_cost += 0.33 * F.grid_sample(cost2d[:, :, :, :, 1],
                                                 sampled_net[:, :, :, 0, torch.Tensor([0, 2]).long()],
                                                 align_corners=True)
            sampled_cost += 0.33 * F.grid_sample(cost2d[:, :, :, :, 2], sampled_net[:, :, :, 0, 1:],
                                                 align_corners=True)
            # maximise probabilities
            loss = (-sampled_cost).mean()
            # minimise diffusion regularisation penalty
            reg_loss = lambda_weight * ((fitted_grid[0, :, :, 1:, :] - fitted_grid[0, :, :, :-1, :]) ** 2).mean() + \
                       lambda_weight * ((fitted_grid[0, :, 1:, :, :] - fitted_grid[0, :, :-1, :, :]) ** 2).mean() + \
                       lambda_weight * ((fitted_grid[0, :, :, :, 1:] - fitted_grid[0, :, :, :, :-1]) ** 2).mean()

            (reg_loss + loss).backward()

            optimizer.step()
    # return both low-resolution and high-resolution transformation
    dense_flow_fit = F.interpolate(fitted_grid.detach(), size=(H, W, D), mode='trilinear', align_corners=True)

    return dense_flow_fit, fitted_grid
