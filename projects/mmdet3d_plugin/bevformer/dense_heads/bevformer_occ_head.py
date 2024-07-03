# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Xiaoyu Tian
# ---------------------------------------------

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.utils import TORCH_VERSION, digit_version

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models import HEADS
from mmdet.models.dense_heads import DETRHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmcv.runner import force_fp32, auto_fp16
from projects.mmdet3d_plugin.models.utils.bricks import run_time
import numpy as np
import mmcv
import cv2 as cv
from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models.builder import build_loss
from mmcv.runner import BaseModule, force_fp32
import math
from projects.mmdet3d_plugin.bevformer.modules.nus_param import MultiClassDiceLoss, CE_ssc_loss, nusc_class_frequencies
# from projects.mmdet3d_plugin.models.utils.dice_loss import MultiClassDiceLoss
# from projects.mmdet3d_plugin.models.modules.nus_param import nusc_class_frequencies
# from projects.mmdet3d_plugin.models.modules.nus_param import geo_scal_loss, sem_scal_loss, CE_ssc_loss

def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

def coords_normalize(coords):
    pc_range=[-40, -40, -1.0, 40, 40, 5.4]
    norm_coords = coords.clone()
    # coords : [n,3]
    x_min, x_max = pc_range[0], pc_range[3]
    y_min, y_max = pc_range[1], pc_range[4]
    z_min, z_max = pc_range[2], pc_range[5]
    norm_coords[...,0] = (norm_coords[...,0] - x_min) / (x_max - x_min)
    norm_coords[...,1] = (norm_coords[...,1] - y_min) / (y_max - y_min)
    norm_coords[...,2] = (norm_coords[...,2] - z_min) / (z_max - z_min)
    return norm_coords

@HEADS.register_module()
class BEVFormerOccHead(BaseModule):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 pc_range=[-40, -40, -1.0, 40, 40, 5.4],
                 bev_h=30,
                 bev_w=30,
                 loss_occ=None,
                 use_mask=False,
                 positional_encoding=None,
                #  feat_positional_encoding=None,
                 # occ as set of points args
                 num_points=10000, # num point of quey
                 eval_num_points = 20000, # eval num points
                 **kwargs):

        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.num_classes=kwargs['num_classes']
        self.use_mask=use_mask

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage


        self.pc_range = pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]
        self.num_cls_fcs = num_cls_fcs - 1

        # occ as set of points asrgs
        self.num_points = num_points
        self.eval_num_points = eval_num_points

        super(BEVFormerOccHead, self).__init__()



        self.loss_occ = build_loss(loss_occ)
        self.dice_loss = MultiClassDiceLoss(self.num_classes)

        self.positional_encoding = build_positional_encoding(
            positional_encoding)

        # self.feat_positional_encoding = build_positional_encoding(
        #     feat_positional_encoding)

        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)

        self._init_layers()

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        # if self.loss_cls.use_sigmoid:
        #     bias_init = bias_init_with_prob(0.01)
        #     for m in self.cls_branches:
        #         nn.init.constant_(m[-1].bias, bias_init)

    def _init_layers(self):
        # from PETR
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.feat_adapt_pos3d = nn.Sequential(
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims, self.embed_dims, kernel_size=1, stride=1, padding=0),
        )
        self.feat_position_encoder = nn.Sequential(
            nn.Linear(3, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            # nn.ReLU(inplace=True),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.embed_dims), 
            nn.LayerNorm(self.embed_dims),
            # nn.ReLU(inplace=True),
            nn.GELU(),
        )

    @auto_fp16(apply_to=('mlvl_feats'))
    def forward(self, mlvl_feats, img_metas, voxel_semantics=None, mask_camera=None, prev_bev=None, only_bev=False, test=False):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder.
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype
        object_query_embeds = None
        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        use_bev_volume = False


        if only_bev:  # only use encoder to obtain BEV features, TODO: refine the workaround
            return self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
                only_bev=only_bev,
            )
        else:
            # occ as set of points train
            if test == False:
                tag = 'train'
                xyz_points_3d , voxel_label, mask_camera_select = self.sample_pts(voxel_semantics, mask_camera)
                xyz_points_norm = coords_normalize(xyz_points_3d) # [n,3] 0~1
                point_query_embeds = self.query_embedding(pos2posemb3d(xyz_points_norm)) # get query [num_points, embed_dim]

                outputs,outputs_list = self.transformer(
                    mlvl_feats,
                    bev_queries, # [bev_h * bev_w, emb_dim]
                    object_query_embeds,
                    self.bev_h,
                    self.bev_w,
                    grid_length=(self.real_h / self.bev_h,
                                self.real_w / self.bev_w),
                    bev_pos=bev_pos,
                    reg_branches=None,  # noqa:E501
                    cls_branches=None,
                    img_metas=img_metas,
                    prev_bev=prev_bev,
                    # occ as set of points
                    # point_query = point_query_embeds,
                    point_norm = xyz_points_norm, # 0~1
                    point_pos=point_query_embeds,
                    only_bev=only_bev,
                    use_bev_volume=use_bev_volume,
                    xyz_points_3d = xyz_points_3d, # not norm
                    query_embedding = self.query_embedding,
                )
                point_outputs = outputs
                # bev_embed, bev_outputs,point_outputs = outputs
                outs = {
                    # 'bev_embed': bev_embed,
                    # 'bev_occ':bev_outputs,
                    'point_occ': point_outputs,
                    'voxel_label_select':voxel_label,
                    'mask_camera_select':mask_camera_select,
                    'use_bev_volume':use_bev_volume,

                }
                return outs
       
            # occ as set of points test
        
            else:
                tag = 'eval'
                # occ_pred_list=[]
                # xyz_points_list = self.sample_pts(voxel_semantics, mask_camera, tag=tag)

                occIdx = np.where(mask_camera[0][0].cpu().numpy())
                # occIdx = (occIdx[0][:8000], occIdx[1][:8000], occIdx[2][:8000])
                voxel_size = [0.4,0.4,0.4]
                points = np.concatenate((occIdx[0][:, None] * voxel_size[0] + voxel_size[0] / 2 + self.pc_range[0], \
                                         occIdx[1][:, None] * voxel_size[1] + voxel_size[1] / 2 + self.pc_range[1], \
                                         occIdx[2][:, None] * voxel_size[2] + voxel_size[2] / 2 + self.pc_range[2]),
                                         axis=1)

                num_groups = points.shape[0] // self.eval_num_points
                # print('points_num',points.shape[0])
                occ_res_zero = np.zeros([200,200,16])
                occ_score_list = []
                occ_adaptive_list=[]
                for i in range(num_groups):
                    start_idx = i * self.eval_num_points
                    end_idx = (i + 1) * self.eval_num_points
                    group_pts = points[start_idx:end_idx]
                    xyz_points_norm = coords_normalize(torch.from_numpy(group_pts).to(voxel_semantics[0].device)).to(torch.float32)
                    point_query_embeds = self.query_embedding(pos2posemb3d(xyz_points_norm))

                    outputs , outputs_list= self.transformer(
                        mlvl_feats,
                        bev_queries,
                        object_query_embeds,
                        self.bev_h,
                        self.bev_w,
                        grid_length=(self.real_h / self.bev_h,
                                    self.real_w / self.bev_w),
                        bev_pos=bev_pos,
                        reg_branches=None,  # noqa:E501
                        cls_branches=None,
                        img_metas=img_metas,
                        prev_bev=prev_bev,
                        # occ as set of points
                        # point_query = point_query_embeds,
                        point_norm = xyz_points_norm,
                        point_pos=point_query_embeds,
                        only_bev=only_bev,
                        test=True,
                        use_bev_volume=use_bev_volume,
                        xyz_points_3d = torch.from_numpy(group_pts).to(voxel_semantics[0].device).to(torch.float32),
                        query_embedding = self.query_embedding,

                    )
                    point_outputs = outputs
                    point_outputs = torch.nan_to_num(point_outputs)
                    occ_score=point_outputs.softmax(-1)
                    occ_score=occ_score.argmax(-1)
                    occ_score_list.append(occ_score[0])
                    # occ_adaptive_list.append(self.adaptive_select(occ_score,outputs_list))
                if points.shape[0] % self.eval_num_points != 0:
                    start_idx = num_groups * self.eval_num_points
                    end_idx = points.shape[0]
                    remaining_data = points[start_idx:end_idx]
                    xyz_points_norm = coords_normalize(torch.from_numpy(remaining_data).to(voxel_semantics[0].device)).to(torch.float32)
                    point_query_embeds = self.query_embedding(pos2posemb3d(xyz_points_norm))
                    outputs, outputs_list = self.transformer(
                        mlvl_feats,
                        bev_queries,
                        object_query_embeds,
                        self.bev_h,
                        self.bev_w,
                        grid_length=(self.real_h / self.bev_h,
                                    self.real_w / self.bev_w),
                        bev_pos=bev_pos,
                        reg_branches=None,  # noqa:E501
                        cls_branches=None,
                        img_metas=img_metas,
                        prev_bev=prev_bev,
                        # occ as set of points
                        # point_query = point_query_embeds,
                        point_norm = xyz_points_norm,
                        point_pos=point_query_embeds,
                        only_bev=only_bev,
                        test=True,
                        use_bev_volume=use_bev_volume,
                        xyz_points_3d = torch.from_numpy(remaining_data).to(voxel_semantics[0].device).to(torch.float32),
                        query_embedding = self.query_embedding,

                        # feat_pos_embed = feat_pos_embed,

                    )
                    point_outputs = outputs
                    point_outputs = torch.nan_to_num(point_outputs)
                    occ_score=point_outputs.softmax(-1)
                    occ_score=occ_score.argmax(-1)
                    occ_score_list.append(occ_score[0])
                    # occ_adaptive_list.append(self.adaptive_select(occ_score,outputs_list))

                # occ_score_cat = torch.cat(occ_adaptive_list).cpu().numpy()

                occ_score_cat = torch.cat(occ_score_list).cpu().numpy()
                occ_res_zero[occIdx] = occ_score_cat
                outs = {
                    # 'bev_embed': bev_embed,
                    'occ_pred':torch.from_numpy(occ_res_zero).to(voxel_semantics[0].device),
                }
                return outs

              
                return outs

    def adaptive_select(self, occ_score, outputs_list):
        num_points = occ_score.shape[1]
        x1 = num_points * 1 // 1000 + 1
        x2 = num_points * 20 // 100
        x3 = num_points - x1 - x2

        output_adaptive = torch.zeros_like(outputs_list[0][0])

        output_score_0 = outputs_list[0].softmax(-1).max(-1)[0][0]
        _, top_indices_0 = output_score_0.topk(x1)
        output_0 = outputs_list[0][0][top_indices_0]
        output_adaptive[top_indices_0] = output_0

        output_score_1 = outputs_list[1].softmax(-1).max(-1)[0][0]
        output_score_1[top_indices_0] = 0
        _, top_indices_1 = output_score_1.topk(x2)
        output_1 = outputs_list[1][0][top_indices_1]
        output_adaptive[top_indices_1] = output_1

        output_score_2 = outputs_list[2].softmax(-1).max(-1)[0][0]
        output_score_2[top_indices_0] = 0
        output_score_2[top_indices_1] = 0
        _, top_indices_2 = output_score_2.topk(x3)
        output_2 =  outputs_list[2][0][top_indices_2]
        output_adaptive[top_indices_2] = output_2

        # print((occ_score[0] == output_adaptive.softmax(-1).argmax(-1)).sum())

        return output_adaptive.softmax(-1).argmax(-1)
    

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             # gt_bboxes_list,
             # gt_labels_list,
             voxel_semantics,
             mask_camera,
             preds_dicts,
             gt_bboxes_ignore=None,
             img_metas=None):

        loss_dict=dict()
        # bev_occ=preds_dicts['bev_occ']
        point_occ=preds_dicts['point_occ']
        voxel_label_select=preds_dicts['voxel_label_select']
        mask_camera_select=preds_dicts['mask_camera_select']

        # assert voxel_semantics.min()>=0 and voxel_semantics.max()<=17
        # if  preds_dicts['use_bev_volume']:            
        #     bev_losses = self.loss_single(voxel_semantics,mask_camera,bev_occ)
        #     loss_dict['bev_loss_ce']=bev_losses['loss_voxel_ce']
        #     loss_dict['bev_loss_dice']=bev_losses['loss_voxel_dice']

        point_losses = self.loss_single(voxel_label_select,mask_camera_select,point_occ)
        loss_dict['point_loss_ce']=point_losses['loss_voxel_ce']
        loss_dict['point_loss_dice']=point_losses['loss_voxel_dice']

        return loss_dict

    def loss_single(self,voxel_semantics,mask_camera,preds):
        voxel_semantics=voxel_semantics.long()
        loss_dict = {}
        if self.use_mask:
            voxel_semantics_reshape = voxel_semantics[mask_camera].reshape(-1)
            preds_reshape = preds[mask_camera].reshape(-1,self.num_classes)
        else:
            voxel_semantics_reshape = voxel_semantics.reshape(-1)
            preds_reshape = preds.reshape(-1,self.num_classes)  

        loss_dice = self.dice_loss(preds_reshape, voxel_semantics_reshape) 
        loss_dict['loss_voxel_dice'] = loss_dice

        preds_reshape = preds_reshape.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(0,4,3,1,2) # bs ,c ,h, w,d
        voxel_semantics_reshape = voxel_semantics_reshape.unsqueeze(0).unsqueeze(0).unsqueeze(0).permute(0,3,1,2)

        class_weights = torch.from_numpy(1 / np.log(nusc_class_frequencies[:19] + 0.001)).to(torch.float32).to((voxel_semantics.device))
        loss_ce = CE_ssc_loss(preds_reshape, voxel_semantics_reshape, class_weights)
        loss_dict['loss_voxel_ce'] = loss_ce

        # if self.use_mask:
        #     voxel_semantics=voxel_semantics.reshape(-1)
        #     preds=preds.reshape(-1,self.num_classes)
        #     mask_camera=mask_camera.reshape(-1)
        #     num_total_samples=mask_camera.sum()
        #     loss_occ=self.loss_occ(preds,voxel_semantics,mask_camera, avg_factor=num_total_samples)
        # else:
        #     voxel_semantics = voxel_semantics.reshape(-1)
        #     preds = preds.reshape(-1, self.num_classes)
        #     loss_occ = self.loss_occ(preds, voxel_semantics,)

        return loss_dict

    @force_fp32(apply_to=('preds'))
    def get_occ(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            predss : occ results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # return self.transformer.get_occ(
        #     preds_dicts, img_metas, rescale=rescale)
        # print(img_metas[0].keys())

        # occ_out=preds_dicts['occ']
        # occ_score=occ_out.softmax(-1)
        # occ_score=occ_score.argmax(-1)

        occ_out=preds_dicts['occ_pred'].unsqueeze(0).detach().cpu()

        return occ_out

    def sample_pts(self, voxel_semantics, mask_camera, tag='train'):
        
        if tag == 'train':
            num_pts_all = self.num_points 

            xyz_points_all = self.infer_query_generate_all(voxel_semantics.device,tag).view(-1,3)
            voxel_label_all = self.select_voxel(voxel_semantics[0],xyz_points_all).view(-1)
            mask_camera_all = self.select_voxel(mask_camera[0],xyz_points_all).view(-1).to(torch.uint8)
            mask_camera_all_neg = 1-mask_camera_all

        
            if mask_camera_all.sum() >= self.num_points:
                xyz_points_cam = xyz_points_all[mask_camera_all]
                shuffled_indices = torch.randperm(xyz_points_cam.size(0))
                xyz_points_shuffle = xyz_points_cam.clone()[shuffled_indices]
                xyz_points = xyz_points_shuffle[:self.num_points]
                voxel_label = self.select_voxel(voxel_semantics[0],xyz_points).view(-1)
                # voxel_label = voxel_label_all[mask_camera_all][:self.num_points]
                mask_camera_select = self.select_voxel(mask_camera[0],xyz_points).view(-1).to(torch.uint8)
                # mask_camera_select = mask_camera_all[:self.num_points]
            else:
                xyz_points_pos = xyz_points_all[mask_camera_all][:mask_camera_all.sum()] # mask_cameral true
                xyz_points_neg = xyz_points_all[mask_camera_all_neg][:(self.num_points-mask_camera_all.sum())]
                xyz_points = torch.cat((xyz_points_pos,xyz_points_neg), dim=0)
                voxel_label = self.select_voxel(voxel_semantics[0],xyz_points).view(-1)
                # voxel_label = voxel_label_all[mask_camera_all][:self.num_points]
                mask_camera_select = self.select_voxel(mask_camera[0],xyz_points).view(-1).to(torch.uint8)

            disturb = True
            if disturb:
                disturbance = 0.38 * torch.rand(size=(xyz_points.shape[0],3)) - 0.19
                xyz_points += disturbance.to(xyz_points.device)


            return xyz_points, voxel_label, mask_camera_select

        else : 
            xyz_points_list = self.infer_query_generate_all(voxel_semantics[0].device,tag)
            # voxel_label_list = [self.select_voxel(voxel_semantics[0][0],xyz.view(-1,3)).view(-1) \
            #     for xyz in xyz_points_list]
            return xyz_points_list

    def infer_query_generate_all(self,device,tag): # 16 * 50 * 50 * 4
        # gt
        scene_size = [200,200,16]
        voxel_size = 0.4
        xyz_grid_ori = torch.stack(torch.meshgrid(
            torch.linspace(self.pc_range[0] + 0.5 * voxel_size , self.pc_range[3] - 0.5 * voxel_size, scene_size[0]),
            torch.linspace(self.pc_range[1] + 0.5 * voxel_size , self.pc_range[4] - 0.5 * voxel_size, scene_size[1]),
            torch.linspace(self.pc_range[2] + 0.5 * voxel_size , self.pc_range[5] - 0.5 * voxel_size, scene_size[2]),
        ), -1)
        # my voxel 
        # num of grid in 3d
        small_grid_list = []
        batch_size = self.eval_num_points
        xyz_grid_ori_review = xyz_grid_ori.clone().view(-1,3)

        for i in range(0, xyz_grid_ori_review.size(0), batch_size):
            small_tensor = xyz_grid_ori_review[i:i+batch_size].to(device)
            small_grid_list.append(small_tensor)
        # for i in range(0, 200, self.x_input):
        #     for j in range(0, 200, self.y_input):
        #         for k in range(0, 16, self.z_input):
        #             small_grid = xyz_grid_ori.clone()[i:i+self.x_input, j:j+self.y_input, k:k+self.z_input,:].to(device)
        #             small_grid_list.append(small_grid)
        # small_grids = xyz_grid_ori.unfold(0, 50, 50).unfold(1, 50,50).unfold(2, 4, 4)
        if tag == "train":
            return xyz_grid_ori.to(device)
        elif tag == "eval":
            return small_grid_list

    def select_voxel(self,voxel_semantics,xyz):
        xyz = xyz.to(voxel_semantics.device)
        scene_size = [200,200,16]
        # self.pc_range [-40, -40, -1, 40, 40, 5.4]
        # [200 ,200 ,16] * 0.4 -> 80 , 80 ,6.4
        # grid coordinate
        x =  (xyz[...,0] - self.pc_range[0]) / (self.pc_range[3] - self.pc_range[0]) * 200 # v1_251改：199 -> 200
        y =  (xyz[...,1] - self.pc_range[1]) / (self.pc_range[4] - self.pc_range[1]) * 200 
        z =  (xyz[...,2] - self.pc_range[2]) / (self.pc_range[5] - self.pc_range[2]) * 16
        x_norm = (2 * x - 200) / 200 # -1 ~ 1
        y_norm = (2 * y - 200) / 200 
        z_norm = (2 * z - 16) / 16
        
        xyz_norm = torch.stack((z_norm, y_norm, x_norm),axis = 1).unsqueeze(0) #1,50,50,2,3

        xyz_norm_reshape = xyz_norm.reshape(1,-1,1,1,3) # 1,5000, 1,1,3

        if voxel_semantics.dim() == 3 :
            voxel_semantics_trans = voxel_semantics.unsqueeze(0).unsqueeze(0).clone()# 1,1,200,200,16
        elif voxel_semantics.dim() == 4 :# [c,200,200,16]
            voxel_semantics_trans = voxel_semantics.unsqueeze(0).clone()# 1,C,200,200,1

        voxel_select_reshape = F.grid_sample(voxel_semantics_trans.to(xyz_norm_reshape.dtype), xyz_norm_reshape, mode='nearest') #  1,2,50,50,3
        return voxel_select_reshape


