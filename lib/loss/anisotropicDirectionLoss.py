import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AnisotropicDirectionLoss(nn.Module):
    def __init__(self, scale=0.01, loss_lambda=2.0, edge_info=None):
        super(AnisotropicDirectionLoss, self).__init__()
        self.max_node_number = 1000
        self.scale = scale
        self.loss_lambda = loss_lambda
        self.neighbors = self._get_neighbors(edge_info)
        self.bins = list()
        self.max_bins = 1000

    def __repr__(self):
        return "AnisotropicDirectionLoss()"

    def _get_neighbors(self, edge_info):
        neighbors = np.arange(self.max_node_number)[:,np.newaxis].repeat(3, axis=1)
        for is_closed, indices in edge_info:
            n = len(indices)
            for i in range(n):
                cur_id = indices[i]
                pre_id = indices[(i-1)%n]
                nex_id = indices[(i+1)%n]
                if not is_closed:
                    if i == 0:
                        pre_id = nex_id
                    elif i == n-1:
                        nex_id = pre_id
                neighbors[cur_id][0] = cur_id
                neighbors[cur_id][1] = pre_id
                neighbors[cur_id][2] = nex_id
        return neighbors

    def _inverse_vector(self, vector):
        """
        input: b x n x 2
        output: b x n x 2
        """
        inversed_vector = torch.stack((-vector[:,:,1], vector[:,:,0]), dim=-1)
        return inversed_vector

    def _get_normals_from_neighbors(self, landmarks):
        # input: b x n x 2
        # output: # b x n x 2
        point_num = landmarks.shape[1]
        itself = self.neighbors[0:point_num, 0]
        previous_neighbors = self.neighbors[0:point_num, 1]
        next_neighbors = self.neighbors[0:point_num, 2]
    
        # condition 1
        bi_normal_vector = F.normalize(landmarks[:, previous_neighbors] - landmarks[:, itself], p=2, dim=-1) + \
                           F.normalize(landmarks[:, next_neighbors] - landmarks[:, itself], p=2, dim=-1)
        # condition 2
        previous_tangent_vector = landmarks[:, previous_neighbors] - landmarks[:, itself]
        next_tangent_vector = landmarks[:, next_neighbors] - landmarks[:, itself]
    
        normal_vector = torch.where(previous_tangent_vector == next_tangent_vector, self._inverse_vector(previous_tangent_vector), bi_normal_vector)
    
        normal_vector = F.normalize(normal_vector, p=2, dim=-1)
        return normal_vector

    def _make_grid(self, h, w):
        yy, xx = torch.meshgrid(
            torch.arange(h).float() / (h-1)*2-1,
            torch.arange(w).float() / (w-1)*2-1)
        return yy, xx

    def _get_loss_lambda(self, pv_gt, normal_force, tangent_force, normal_vector, tangent_vector, heatmap=None, landmarks=None, lambda_mode=3):
        # fix
        if lambda_mode == 1:
            # 1
            loss_lambda = self.loss_lambda
        # dynamic
        elif lambda_mode == 2:
            loss_lambda = torch.clamp(tangent_force.pow(2) / torch.clamp(normal_force.pow(2), min=1e-6), min=1.0, max=9.0)
            # b x n
            loss_lambda = loss_lambda.detach()
        # heatmap-based
        elif lambda_mode == 3:
            batch, npoints, h, w = heatmap.shape

            weightmap = heatmap / torch.clamp(heatmap.sum([2, 3], keepdim=True), min=1e-6) # b x n x h x w

            yy, xx = self._make_grid(h, w)
            yy = yy.view(1, 1, h, w).to(heatmap) # 1 x 1 x h x w
            xx = xx.view(1, 1, h, w).to(heatmap) # 1 x 1 x h x w
            coordinate = torch.stack([xx, yy], dim=-3) # 1 x 1 x 2 x h x w

            # landmarks
            if False:
                yy_coord = (yy * weightmap).sum([2, 3], keepdim=True) # b x n x 1 x 1
                xx_coord = (xx * weightmap).sum([2, 3], keepdim=True) # b x n x 1 x 1
                landmarks = torch.stack([xx_coord, yy_coord], dim=2) # b x n x 2 x 1 x 1
            else:
                landmarks = landmarks.unsqueeze(-1).unsqueeze(-1)

            direction = coordinate - landmarks # b x n x 2 x h x w
            #direction = F.normalize(direction, p=2, dim=2)

            # normal and tangent
            if False:
                dx = torch.mul(direction[:, :, 0], weightmap) # b x n x h x w
                dy = torch.mul(direction[:, :, 1], weightmap) # b x n x h x w
                dx = dx * dy.sign()
                dy = dy.abs()
                dx = dx.sum([2, 3]) # b x n
                dy = dy.sum([2, 3]) # b x n
                tangent_vector = torch.stack([dx, dy], dim=-1) # b x n x 2
                tangent_vector = F.normalize(tangent_vector, p=2, dim=-1) # b x n x 2
                normal_vector = self._inverse_vector(tangent_vector) # b x n x 2

            # coordinate: 1 x 1 x 2 x h x w
            # landmarks: b x n x 2 x 1 x 1
            # direction: b x n x 2 x h x w
            # normal_vector: b x n x 2
            # tangent_vector: b x n x 2
            normal_std2 = torch.mul(direction, normal_vector.unsqueeze(-1).unsqueeze(-1)).sum(dim=2, keepdim=False).pow(2) * weightmap # b x n x h x w
            tangent_std2 = torch.mul(direction, tangent_vector.unsqueeze(-1).unsqueeze(-1)).sum(dim=2, keepdim=False).pow(2) * weightmap # b x n x h x w

            loss_lambda = torch.clamp(tangent_std2.sum([2, 3]) / torch.clamp(normal_std2.sum([2, 3]), min=1e-6), min=1.0, max=9.0) # b x n
            # b x n
            loss_lambda = loss_lambda.detach()
        # statistic
        elif lambda_mode == 4:
            cur_loss_lambda = tangent_force.pow(2) / torch.clamp(normal_force.pow(2), min=1e-6) # b x n
            self.bins.extend(cur_loss_lambda.tolist()) # (1000 x b) x n
            while len(self.bins) > self.max_bins:
                del self.bins[0]
            loss_lambda = torch.tensor(self.bins).to(pv_gt) # (1000 x b) x n
            loss_lambda = loss_lambda.mean(dim=0, keepdim=True) # 1 x n
            loss_lambda = torch.clamp(loss_lambda, min=1.0, max=9.0)
            # 1 x n
            loss_lambda = loss_lambda.detach()
        # statistic
        elif lambda_mode == 5:
            self.bins.extend(pv_gt.tolist()) # (1000 x b) x n x 2
            while len(self.bins) > self.max_bins:
                del self.bins[0]
            direction = torch.tensor(self.bins).to(pv_gt) # (1000 x b) x n x 2
            dx = direction[:, :, 0] # (1000 x b) x n
            dy = direction[:, :, 1] # (1000 x b) x n
            dx = dx * dy.sign() # (1000 x b) x n
            dy = dy.abs() # (1000 x b) x n
            dx = dx.sum([0]) # n
            dy = dy.sum([0]) # n
            tangent_vector = torch.stack([dx, dy], dim=-1) # n x 2
            tangent_vector = F.normalize(tangent_vector, p=2, dim=-1) # n x 2
            normal_vector = torch.stack((-tangent_vector[:,1], tangent_vector[:,0]), dim=-1) # n x 2

            normal_std2 = torch.mul(direction, normal_vector.unsqueeze(0)).sum(dim=-1).pow(2).sum(dim=0) # n
            tangent_std2 = torch.mul(direction, tangent_vector.unsqueeze(0)).sum(dim=-1).pow(2).sum(dim=0) # n

            loss_lambda = torch.clamp(tangent_std2 / torch.clamp(normal_std2, min=1e-6), min=1.0, max=9.0).unsqueeze(0) # 1 x n
            # 1 x n
            loss_lambda = loss_lambda.detach()
        else:
            assert False
        return loss_lambda


    def forward(self, output, groundtruth, heatmap=None, landmarks=None):
        """
            input:  b x n x 2
            output: b x n x 1
        """
        normal_vector = self._get_normals_from_neighbors(groundtruth) # b x n x 2, [-1, 1]
        tangent_vector = self._inverse_vector(normal_vector) # b x n x 2, [-1, 1]

        pv_gt = output - groundtruth # b x n x 2, [-1, 1]

        normal_force = torch.mul(pv_gt, normal_vector).sum(dim=-1, keepdim=False) # b x n
        tangent_force = torch.mul(pv_gt, tangent_vector).sum(dim=-1, keepdim=False) # b x n
        
        loss_lambda = self._get_loss_lambda(pv_gt.detach(), normal_force.detach(), tangent_force.detach(), normal_vector.detach(), tangent_vector.detach(), heatmap.detach(), landmarks.detach())

        alpha = 2 * loss_lambda / (loss_lambda + 1.0)
        belta = 2 * 1 / (loss_lambda + 1.0)
        delta_2_asy = alpha * normal_force.pow(2) + belta * tangent_force.pow(2) # b x n

        delta_2_sy = pv_gt.pow(2).sum(dim=-1, keepdim=False) # b x n

        delta_2 = torch.where(normal_vector.norm(p=2, dim=-1) < 0.5, delta_2_sy, delta_2_asy)

        delta = delta_2.clamp(min=1e-128).sqrt() # delta_2.sqrt()
        loss = torch.where(delta < self.scale, 0.5 / self.scale * delta_2, delta - 0.5 * self.scale)

        return loss.mean()
