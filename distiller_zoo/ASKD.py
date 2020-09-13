from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassSimilarityLoss(nn.Module):
    def __init__(self):
        super(ClassSimilarityLoss, self).__init__()

    def forward(self, y_s, y_t):
        y_t = y_t.unsqueeze(2)
        y_s = y_s.unsqueeze(2)
        # distance loss
        with torch.no_grad():
            t_d = self.pdist(y_t)  # (N, m, m)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td
        d = self.pdist(y_s)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)

        return loss_d

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2)  # (N, m, 1)
        prod = e @ e.permute(0, 2, 1)  # (N, m, m)
        res = (e_square + e_square.permute(0, 2, 1) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[:, range(e.size(1)), range(e.size(1))] = 0
        return res


class RegionalSimilarityLoss(nn.Module):

    def __init__(self, r):
        super(RegionalSimilarityLoss, self).__init__()
        self.r = r

    def forward(self, g_s, g_t):
        loss = [self.region_loss(f_s, f_t) for f_s, f_t in zip(g_s, g_t)]
        return loss

    def region_loss(self, f_s, f_t):
        assert f_s.size(1) == f_t.size(1), 'error: unequal channels'
        N, C, H_t, W_t = f_t.size()
        N, C, H_s, W_s = f_s.size()
        assert H_t == W_t and H_s == W_s, 'error: unequal W and H in feature maps'

        if self.r > min(H_t, H_s):
            self.r = min(H_t, H_s)

        # Region division
        regions_t = F.avg_pool2d(f_t, H_t // self.r).view(N, C, -1)  # (N, C, r*r)
        regions_s = F.avg_pool2d(f_s, H_s // self.r).view(N, C, -1)  # (N, C, r*r)

        # Regional similarity
        with torch.no_grad():
            t_d = self.pdist(regions_t, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td  # normalize
        d = self.pdist(regions_s, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d
        loss_d = F.smooth_l1_loss(d, t_d)
        return loss_d

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        # e: (N, C, r*r)
        e = e[..., None]
        e_square = e.pow(2)  # (N, C, r*r, 1)
        prod = e @ e.permute(0, 1, 3, 2)  # (N, C, r*r, r*r)
        res = (e_square + e_square.permute(0, 1, 3, 2) - 2 * prod).clamp(min=eps)  # (N, C, r*r, r*r)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[:, :, range(e.size(2)), range(e.size(2))] = 0
        return res
