import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureNet(nn.Module):
    def __init__(self):
        super(FeatureNet, self).__init__()
        # TODO
        self.relu = nn.ReLU()
        self.conv8_1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv8_2 = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.convBlock1 = nn.Sequential(
            self.conv8_1,
            nn.BatchNorm2d(8),
            self.relu,
            self.conv8_2,
            nn.BatchNorm2d(8),
            self.relu
        )

        self.conv16_1 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.conv16_2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv16_3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.convBlock2 = nn.Sequential(
            self.conv16_1,
            nn.BatchNorm2d(16),
            self.relu,
            self.conv16_2,
            nn.BatchNorm2d(16),
            self.relu,
            self.conv16_3,
            nn.BatchNorm2d(16),
            self.relu
        )

        self.conv32_1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.conv32_2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv32_3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.convBlock3 = nn.Sequential(
            self.conv32_1,
            nn.BatchNorm2d(32),
            self.relu,
            self.conv32_2,
            nn.BatchNorm2d(32),
            self.relu,
            self.conv32_3,
            nn.BatchNorm2d(32),
            self.relu
        )

        self.convLast = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: [B,3,H,W]
        # TODO
        out = self.convBlock1(x)
        out2 = self.convBlock2(out)
        out3 = self.convBlock3(out2)
        return self.convLast(out3)


class SimlarityRegNet(nn.Module):
    def __init__(self, G):
        super(SimlarityRegNet, self).__init__()
        # TODO
        self.conv1 = nn.Conv2d(in_channels=G, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1)
        self.upconv1 = nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1,
                                          output_padding=1)
        self.convout = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B,G,D,H,W]
        # out: [B,D,H,W]
        # TODO
        B, G, D, H, W = x.size()
        s = x.transpose(1, 2).reshape(B * D, G, H, W)
        c0 = self.relu(self.conv1(s))
        c1 = self.relu(self.conv2(c0))
        c2 = self.relu(self.conv3(c1))
        c3 = self.upconv1(c2)
        # c3_cat = torch.cat([c3, c1], dim=1)
        c3_cat = c3 + c1
        c4 = self.upconv2(c3_cat)
        # c4_cat = torch.cat([c4, c0], dim=1)
        c4_cat = c4 + c0
        s_reg = self.convout(c4_cat)  # [B*D,1,H,W]
        return s_reg.reshape(B, D, H, W)


def warping(src_fea, src_proj, ref_proj, depth_values):
    # src_fea: [B, C, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # depth_values: [B, D]
    # out: [B, C, D, H, W]
    B, C, H, W = src_fea.size()
    D = depth_values.size(1)
    # compute the warped positions with depth values
    with torch.no_grad():
        # relative transformation from reference to source view
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]
        y, x = torch.meshgrid([torch.arange(0, H, dtype=torch.float32, device=src_fea.device),
                               torch.arange(0, W, dtype=torch.float32, device=src_fea.device)])
        y, x = y.contiguous(), x.contiguous()  # [H,W]
        y, x = y.view(H * W), x.view(H * W)
        # TODO
        src_coords = torch.cat([x.unsqueeze(0), y.unsqueeze(0), torch.ones(1, H * W, dtype=torch.float32).to(src_fea.device)],
                               dim=0)  # [3,h*w]
        del x,y
        src_depth = torch.einsum("bd, sn -> bdsn", depth_values, src_coords)  # [B,D,3,H*W]
        del src_coords
        rot_exp = rot.unsqueeze(1).expand(B, D, 3, 3)  # [B,D,3,3]
        trans_exp = trans.unsqueeze(1).expand(B, D, 3, 1)  # [B,D,3,1]
        warped_coords = torch.matmul(rot_exp, src_depth) + trans_exp  # [B,D,3,H*W]
        del src_depth
        # Normalize to image coords
        warped_img_coords = warped_coords / warped_coords[:, :, -1:, :].expand(B, D, 3, H * W)
        warped_grid = torch.transpose(warped_img_coords, -2, -1).reshape(B, D, H, W, -1)  # [B,D,H,W,3]
        # Normalize warped_grid to [-1,1]
        warped_x = warped_grid[..., 0:1] / (W - 1.) * 2 - 1.
        warped_y = warped_grid[..., 1:2] / (H - 1.) * 2 - 1.
        del warped_grid, warped_img_coords, warped_coords
        warped_grid_norm = torch.cat([warped_x, warped_y], dim=-1)
        del warped_x, warped_y
    # get warped_src_fea with bilinear interpolation (use 'grid_sample' function from pytorch)
    # TODO
    warped_src_fea = F.grid_sample(src_fea, warped_grid_norm.reshape(B, D * H, W, -1),
                                   mode="bilinear")
    warped_src_fea = warped_src_fea.reshape(B, C, D, H, W)
    return warped_src_fea


def group_wise_correlation(ref_fea, warped_src_fea, G):
    # ref_fea: [B,C,H,W]
    # warped_src_fea: [B,C,D,H,W]
    # out: [B,G,D,H,W]
    # TODO
    B, C, D, H, W = warped_src_fea.size()
    ref_fea_group = ref_fea.reshape(B, G, -1, H, W)
    warped_src_fea_group = warped_src_fea.reshape(B, G, -1, D, H, W)
    similarity = torch.sum(ref_fea_group.unsqueeze(3).expand(B, G, -1, D, H, W) * warped_src_fea_group, dim=2) / (
            C / G)  # [B,G,D,H,W]
    return similarity


def depth_regression(p, depth_values):
    # p: probability volume [B, D, H, W]
    # depth_values: discrete depth values [B, D]
    # TODO
    B, D, H, W = p.size()
    depth_reg = torch.sum(depth_values[:, :, None, None].expand(B, D, H, W) * p, dim=1)  # [B,H,W]
    return depth_reg


def mvs_loss(depth_est, depth_gt, mask):
    # depth_est: [B,1,H,W]
    # depth_gt: [B,1,H,W]
    # mask: [B,1,H,W]
    # TODO
    l1_loss = nn.L1Loss(reduction="mean")
    mask_idx = mask.bool()
    input = depth_est[mask_idx]
    target = depth_gt[mask_idx]
    return l1_loss(input, target)
