# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import torch
import torch.nn as nn

from pytorch3d.structures import Pointclouds

EPS = 1e-2
PI = 3.141592653589793

def get_splatter(
    name, depth_values, opt=None, size=256, C=64, points_per_pixel=8
):
    if name == "xyblending":
        from models.layers.z_buffer_layers import RasterizePointsXYsBlending

        return RasterizePointsXYsBlending(
            C,
            learn_feature=opt.learn_default_feature,
            radius=opt.radius,
            size=size,
            points_per_pixel=points_per_pixel,
            opts=opt,
        )
    else:
        raise NotImplementedError()


class PtsManipulator(nn.Module):
    def __init__(self, W, C=64, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=W, C=C, points_per_pixel=opt.pp_pixel
        )

        xs = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1
        ys = torch.linspace(0, W - 1, W) / float(W - 1) * 2 - 1

        xs = xs.view(1, 1, 1, W).repeat(1, 1, W, 1)
        ys = ys.view(1, 1, W, 1).repeat(1, 1, 1, W)

        xyzs = torch.cat(
            (xs, -ys, -torch.ones(xs.size()), torch.ones(xs.size())), 1
        ).view(1, 4, -1)

        self.register_buffer("xyzs", xyzs)

    def project_pts(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)

        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj = K.bmm(wrld_X)

        # And finally we project to get the final result
        mask = (xy_proj[:, 2:3, :].abs() < EPS).detach()

        # Remove invalid zs that cause nans
        zs = xy_proj[:, 2:3, :]
        zs[mask] = EPS

        sampler = torch.cat((xy_proj[:, 0:2, :] / -zs, xy_proj[:, 2:3, :]), 1)
        sampler[mask.repeat(1, 3, 1)] = -10
        # Flip the ys
        sampler = sampler * torch.Tensor([1, -1, -1]).unsqueeze(0).unsqueeze(
            2
        ).to(sampler.device)

        return sampler

    def forward_justpts(
        self, src, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
        result = self.splatter(pointcloud, src)

        return result

    def forward(
        self,
        alphas,
        src,
        pred_pts,
        K,
        K_inv,
        RT_cam1,
        RTinv_cam1,
        RT_cam2,
        RTinv_cam2,
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)
            alphas = alphas.view(bs, 1, -1).permute(0, 2, 1).contiguous()

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        result = self.splatter(pts3D.permute(0, 2, 1).contiguous(), alphas, src)

        return result



class EquiPtsManipulator(nn.Module):
    def __init__(self, H, W, C=64, opt=None):
        super().__init__()
        self.opt = opt

        self.splatter = get_splatter(
            opt.splatter, None, opt, size=(H, W), C=C, points_per_pixel=opt.pp_pixel
        )

        batch_size = 1
        equ_w = W
        equ_h = H
        cen_x = (equ_w - 1) / 2.0
        cen_y = (equ_h - 1) / 2.0
        theta = (2 * (torch.arange(equ_w) - cen_x) / equ_w) * PI
        phi = (2 * (torch.arange(equ_h) - cen_y) / equ_h) * (PI / 2)
        theta = theta[None, :].repeat(equ_h, 1)
        phi = phi[None, :].repeat(equ_w, 1).T

        x = (torch.cos(phi) * torch.sin(theta)).unsqueeze(0).unsqueeze(0)
        y = (torch.sin(phi)).unsqueeze(0).unsqueeze(0)
        z = (torch.cos(phi) * torch.cos(theta)).unsqueeze(0).unsqueeze(0)
        xyzs = torch.cat([x, y, z, torch.ones(x.size())], 1).repeat(batch_size, 1, 1, 1).view(1, 4, -1)
        # xyz: 1 x 4 x (H*W)

        self.register_buffer("xyzs", xyzs)

    def project_pts(
        self, pts3D, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # PERFORM PROJECTION
        # Project the world points into the new view
        # self.xyzs: 1 x 4 x H x W
        # pts3D: batch x 1 x H x W
        projected_coors = self.xyzs * pts3D
        projected_coors[:, -1, :] = 1

        # Transform into camera coordinate of the first view
        cam1_X = K_inv.bmm(projected_coors)

        # Transform into world coordinates
        RT = RT_cam2.bmm(RTinv_cam1)

        wrld_X = RT.bmm(cam1_X)

        # And intrinsics
        xy_proj = K.bmm(wrld_X)

        # Project, xy_proj: batch x 4 x H x W
        normXY = torch.sqrt(xy_proj[:, 0, :]**2 + xy_proj[:, 2, :]**2)
        normXY[(normXY < 1e-6).detach()] = 1e-6
        normXYZ = torch.sqrt(xy_proj[:, 0, :]**2 + xy_proj[:, 1, :]**2 + xy_proj[:, 2, :]**2)
        mask = (normXYZ < EPS).detach()
        normXYZ[mask] = EPS
        u = torch.asin(xy_proj[:, 0, :]/normXY)
        v = torch.asin(xy_proj[:, 1, :]/normXYZ)
        valid = (xy_proj[:, 2, :] < 0) & (u >= 0)
        u[valid] = PI - u[valid]
        valid = (xy_proj[:, 2, :] < 0) & (u < 0)
        u[valid] = -(PI + u[valid])
        u /= PI
        v /= PI/2
        sampler = torch.cat(((u/-1*2).unsqueeze(0), (v/-1).unsqueeze(0), normXYZ.unsqueeze(0)), 1) # batch x 3 x(H*W)
        mask = mask.unsqueeze(0)
        
        sampler[mask.repeat(1, 3, 1)] = -10

        # Flip the ys
        sampler = sampler * torch.Tensor([1, -1, 1]).unsqueeze(0).unsqueeze(
            2
        ).to(sampler.device)

        return sampler

    def forward_justpts(
        self, src, pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        pointcloud = pts3D.permute(0, 2, 1).contiguous()
        result = self.splatter(pointcloud, src)

        return result

    def forward(
        self,
        alphas,
        src,
        pred_pts,
        K,
        K_inv,
        RT_cam1,
        RTinv_cam1,
        RT_cam2,
        RTinv_cam2,
    ):
        # Now project these points into a new view
        bs, c, w, h = src.size()

        if len(pred_pts.size()) > 3:
            # reshape into the right positioning
            pred_pts = pred_pts.view(bs, 1, -1)
            src = src.view(bs, c, -1)
            alphas = alphas.view(bs, 1, -1).permute(0, 2, 1).contiguous()

        pts3D = self.project_pts(
            pred_pts, K, K_inv, RT_cam1, RTinv_cam1, RT_cam2, RTinv_cam2
        )
        result = self.splatter(pts3D.permute(0, 2, 1).contiguous(), alphas, src)

        return result
