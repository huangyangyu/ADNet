import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.blurpool import BlurPool
from .core.coord_conv import CoordConvTh


def _make_grid(h, w):
    yy, xx = torch.meshgrid(
        torch.arange(h).float() / (h-1)*2-1,
        torch.arange(w).float() / (w-1)*2-1)
    return yy, xx


def get_coords_from_heatmap(heatmap):
    """
    inputs:
    - heatmap: batch x npoints x h x w

    outputs:
    - coords: batch x npoints x 2 (x,y), [-1, +1]
    - radius_sq: batch x npoints
    """
    batch, npoints, h, w = heatmap.shape

    yy, xx = _make_grid(h, w)
    yy = yy.view(1, 1, h, w).to(heatmap)
    xx = xx.view(1, 1, h, w).to(heatmap)

    heatmap_sum = torch.clamp(heatmap.sum([2, 3]), min=1e-6)

    yy_coord = (yy * heatmap).sum([2, 3]) / heatmap_sum # batch x npoints
    xx_coord = (xx * heatmap).sum([2, 3]) / heatmap_sum # batch x npoints
    coords = torch.stack([xx_coord, yy_coord], dim=-1)

    return coords


class Activation(nn.Module):
    def __init__(self, kind: str = 'relu', channel=None):
        super().__init__()
        self.kind = kind

        if '+' in kind:
            norm_str, act_str = kind.split('+')
        else:
            norm_str, act_str = 'none', kind

        self.norm_fn = {
            'in': F.instance_norm,
            'bn': nn.BatchNorm2d(channel),
            'bn_noaffine': nn.BatchNorm2d(channel, affine=False, track_running_stats=True),
            'none': None
        }[norm_str]

        self.act_fn = {
            'relu': F.relu,
            'softplus': nn.Softplus(),
            'exp': torch.exp,
            'sigmoid': torch.sigmoid,
            'tanh': torch.tanh,
            'none': None
        }[act_str]

        self.channel = channel

    def forward(self, x):
        if self.norm_fn is not None:
            x = self.norm_fn(x)
        if self.act_fn is not None:
            x = self.act_fn(x)
        return x

    def extra_repr(self):
        return f'kind={self.kind}, channel={self.channel}'


class ConvBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, groups=1):
        super(ConvBlock, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size,
                              stride, padding=(kernel_size-1)//2, groups=groups, bias=True)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU()
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class MultiViewBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, groups=1):
        super(MultiViewBlock, self).__init__()

        assert out_dim % 4 == 0
        dim1 = out_dim // 2
        dim2 = out_dim // 4

        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, dim1, 3, relu=False)
        self.bn2 = nn.BatchNorm2d(dim1)
        self.conv2 = ConvBlock(dim1, dim2, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(dim2)
        self.conv3 = ConvBlock(dim2, dim2, 3, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x

        # inp_dim x mid_dim
        out1 = self.bn1(x)
        out1 = self.relu(out1)
        out1 = self.conv1(out1)

        # mid_dim x mid_dim
        out2 = self.bn2(out1)
        out2 = self.relu(out2)
        out2 = self.conv2(out2)

        # mid_dim x out_dim
        out3 = self.bn3(out2)
        out3 = self.relu(out3)
        out3 = self.conv3(out3)

        out = torch.cat([out1, out2, out3], dim=1)

        out += residual
        return out


class ResBlock(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim=None):
        super(ResBlock, self).__init__()
        if mid_dim is None:
            mid_dim = out_dim // 2
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(inp_dim)
        self.conv1 = ConvBlock(inp_dim, mid_dim, 1, relu=False)
        self.bn2 = nn.BatchNorm2d(mid_dim)
        self.conv2 = ConvBlock(mid_dim, mid_dim, 3, relu=False)
        self.bn3 = nn.BatchNorm2d(mid_dim)
        self.conv3 = ConvBlock(mid_dim, out_dim, 1, relu=False)
        self.skip_layer = ConvBlock(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True

    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out


class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0, up_mode='nearest', 
                 add_coord=False, pool_type="origin", use_multiview=False, 
                 first_one=False, x_dim=64, y_dim=64):
        super(Hourglass, self).__init__()
        nf = f + increase

        if use_multiview:
            Block = MultiViewBlock
        else:
            Block = ResBlock

        if add_coord:
            self.coordconv = CoordConvTh(x_dim=x_dim, y_dim=y_dim,
                                         with_r=True, with_boundary=True,
                                         relu=False, bn=False,
                                         in_channels=f, out_channels=f,
                                         first_one=first_one,
                                         kernel_size=1,
                                         stride=1, padding=0)
        else:
            self.coordconv = None
        self.up1 = Block(f, f)

        # Lower branch
        if pool_type == "origin":
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "blur":
            self.pool1 = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 
                                         BlurPool(f, filt_size=3, stride=2)])
            #self.pool1 = BlurPool(f, filt_size=3, stride=2)
        else:
            assert False

        self.low1 = Block(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n=n-1, f=nf, increase=increase, up_mode=up_mode, add_coord=False, pool_type=pool_type)
        else:
            self.low2 = Block(nf, nf)
        self.low3 = Block(nf, f)
        self.up2 = nn.Upsample(scale_factor=2, mode=up_mode)

    def forward(self, x, heatmap=None):
        if self.coordconv is not None:
            x = self.coordconv(x, heatmap)
        up1 = self.up1(x)
        pool1 = self.pool1(x)
        low1 = self.low1(pool1)
        low2 = self.low2(low1)
        low3 = self.low3(low2)
        up2 = self.up2(low3)
        return up1 + up2


class E2HTransform(nn.Module):
    def __init__(self, edge_info, num_points, num_edges):
        super().__init__()

        e2h_matrix = np.zeros([num_points, num_edges])
        for edge_id, isclosed_indices in enumerate(edge_info):
            is_closed, indices = isclosed_indices
            for point_id in indices:
                e2h_matrix[point_id, edge_id] = 1
        e2h_matrix = torch.from_numpy(e2h_matrix).float()

        # pn x en x 1 x 1.
        self.register_buffer('weight', e2h_matrix.view(
            e2h_matrix.size(0), e2h_matrix.size(1), 1, 1))

        # some keypoints are not coverred by any edges,
        # in these cases, we must add a constant bias to their heatmap weights.
        bias = ((e2h_matrix @ torch.ones(e2h_matrix.size(1)).to(
            e2h_matrix)) < 0.5).to(e2h_matrix)
        # pn x 1.
        self.register_buffer('bias', bias)

    def forward(self, edgemaps):
        # input: batch_size x en x hw x hh.
        # output: batch_size x pn x hw x hh.
        return F.conv2d(edgemaps, weight=self.weight, bias=self.bias)


class StackedHGNetV1(nn.Module):
    def __init__(self, classes_num, edge_info,
                 nstack=4, nlevels=4, in_channel=256, increase=0,
                 add_coord=True, pool_type="origin", use_multiview=False):
        super(StackedHGNetV1, self).__init__()

        self.nstack = nstack
        self.add_coord = add_coord
        self.pool_type = pool_type
        self.use_multiview = use_multiview

        self.num_heats = classes_num[0]
        self.num_edges = classes_num[1]
        self.num_points = classes_num[2]
        self.e2h_transform = E2HTransform(edge_info, self.num_points, self.num_edges)

        if self.add_coord:
            convBlock = CoordConvTh(x_dim=256, y_dim=256,
                                    with_r=True, with_boundary=False,
                                    relu=True, bn=True,
                                    in_channels=3, out_channels=64,
                                    kernel_size=7,
                                    stride=2, padding=3)
        else:
            convBlock = ConvBlock(3, 64, 7, 2, bn=True, relu=True)

        if pool_type == "origin":
            pool = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool_type == "blur":
            pool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1), 
                                   BlurPool(128, filt_size=3, stride=2)])
            #pool = BlurPool(128, filt_size=3, stride=2)
        else:
            assert False

        if self.use_multiview:
            Block = MultiViewBlock
        else:
            Block = ResBlock

        self.pre = nn.Sequential(
            convBlock,
            Block(64, 128),
            pool,
            Block(128, 128),
            Block(128, in_channel)
        )

        self.hgs = nn.ModuleList(
            [Hourglass(n=nlevels, f=in_channel, increase=increase, add_coord=self.add_coord, pool_type=self.pool_type, use_multiview=self.use_multiview, first_one=(_==0))
             for _ in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Block(in_channel, in_channel),
                ConvBlock(in_channel, in_channel, 1, bn=True, relu=True)
            ) for _ in range(nstack)])

        self.out_heatmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_heats, 1, relu=False, bn=False)
             for _ in range(nstack)])
        self.out_edgemaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_edges, 1, relu=False, bn=False)
             for _ in range(nstack)])
        self.out_pointmaps = nn.ModuleList(
            [ConvBlock(in_channel, self.num_points, 1, relu=False, bn=False)
             for _ in range(nstack)])

        self.merge_features = nn.ModuleList(
            [ConvBlock(in_channel, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.merge_heatmaps = nn.ModuleList(
            [ConvBlock(self.num_heats, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.merge_edgemaps = nn.ModuleList(
            [ConvBlock(self.num_edges, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.merge_pointmaps = nn.ModuleList(
            [ConvBlock(self.num_points, in_channel, 1, relu=False, bn=False)
             for _ in range(nstack-1)])
        self.nstack = nstack

        self.heatmap_act = Activation("in+relu", self.num_heats)
        self.edgemap_act = Activation("sigmoid", self.num_edges)
        self.pointmap_act = Activation("sigmoid", self.num_points)

        self.inference = False

    def set_inference(self, inference):
        self.inference = inference

    def get_preds_fromhm(self, hm):
        _, idx = torch.max(hm.view(hm.size(0), hm.size(1), -1), 2)
        idx += 1
        preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()

        preds[:, :, 0] = (preds[:, :, 0] - 1) % hm.size(3)
        preds[:, :, 1] = torch.floor((preds[:, :, 1] - 1) / hm.size(3))

        for i in range(preds.size(0)):
            for j in range(preds.size(1)):
                hm_ = hm[i, j, :]
                pX, pY = int(preds[i, j, 0]), int(preds[i, j, 1])
                if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                    diff = torch.FloatTensor(
                        [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                         hm_[pY + 1, pX] - hm_[pY - 1, pX]]).to(preds)
                    preds[i, j] += diff.sign_() * .25

        preds = (preds * 2 + 1) / torch.tensor([64, 64]).to(preds).view(1, 1, 2) - 1
        return preds

    def forward(self, x):
        x = self.pre(x)

        y = []
        heatmaps = None
        for i in range(self.nstack):
            hg = self.hgs[i](x, heatmap=heatmaps)
            feature = self.features[i](hg)

            heatmaps0 = self.out_heatmaps[i](feature)
            heatmaps = self.heatmap_act(heatmaps0)
            edgemaps0 = self.out_edgemaps[i](feature)
            edgemaps = self.edgemap_act(edgemaps0)
            pointmaps0 = self.out_pointmaps[i](feature)
            pointmaps = self.pointmap_act(pointmaps0)

            edge_point_attention_mask = self.e2h_transform(edgemaps) * pointmaps
            landmarks = get_coords_from_heatmap(edge_point_attention_mask * heatmaps)

            if i < self.nstack - 1:
                x = x + \
                    self.merge_features[i](feature) + \
                    self.merge_heatmaps[i](heatmaps) + \
                    self.merge_edgemaps[i](edgemaps) + \
                    self.merge_pointmaps[i](pointmaps)

            y.append(landmarks)
            y.append(edgemaps)
            y.append(pointmaps)

        return y, edge_point_attention_mask, landmarks
