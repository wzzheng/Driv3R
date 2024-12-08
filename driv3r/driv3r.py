import torch.nn as nn
import torch, cv2, os
import torchvision
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import torchvision

from dust3r.post_process import estimate_focal_knowing_depth
from dust3r.utils.geometry import inv, geotrf

from third_party.raft import load_RAFT
flow_ckpt = "RAFT/models/Tartan-C-T-TSKH-spring540x960-M.pth"

from sam2.build_sam import build_sam2_video_predictor
sam2_checkpoint = "third_party/sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


class DepthBasedWarping(nn.Module):

    def __init__(self, eps=1e-6) -> None:
        super().__init__()
        self.eps = eps

    def generate_grid(self, B, P, H, W, device):
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
        )
        self.coord = torch.ones([B, P, 3, H, W], device=device, dtype=torch.float32)
        self.coord[:, :, 0, ...] = xx
        self.coord[:, :, 1, ...] = yy
        self.coord = self.coord.reshape([B, P, 3, H * W]) # [B, P, 3, M]

    def forward(
        self,
        fir_extrinsics, 
        fir_intrinsics,
        sec_extrinsics, 
        sec_intrinsics,
        fir_depths,
        use_depth=False
    ):
        """
            fir_extrinsics: [B, P, 4, 4]
            fir_intrinsics: [B, P, 4, 4]
            sec_extrinsics: [B, P, 4, 4]
            sec_intrinsics: [B, P, 4, 4]
            fir_depths: [B, P, H, W]
            return: [B, P, 2, H, W]
        """
        B, P, H, W = fir_depths.shape
        device = fir_depths.device
        self.generate_grid(B, P, H, W, device=device)

        fir_sec_transform = torch.bmm(
            torch.inverse(sec_extrinsics.view(-1, 4, 4)),
            fir_extrinsics.view(-1, 4, 4)
        )
        full_projection = torch.bmm(
            sec_intrinsics.view(-1, 4, 4),
            torch.bmm(
                fir_sec_transform,
                torch.inverse(fir_intrinsics.view(-1, 4, 4))
            )
        ) # [B * P, 4, 4]
        fir_depths = fir_depths.reshape(B, P, 1, H * W)

        if use_depth:
            first_coord = fir_depths * self.coord
            first_coord_homo = torch.cat([first_coord, torch.ones_like(fir_depths)], dim=2).view(-1, 4, H * W)
            
            second_coord_homo = torch.bmm(full_projection, first_coord_homo).view(B, P, 4, H * W)
            second_coords = second_coord_homo[:, :, :2, :] / (second_coord_homo[:, :, 2:3, :] + self.eps)
            
            second_coords = second_coords.view(B, P, 2, H, W)
            sec_grid = self.coord[:, :, :2, :].view(B, P, 2, H, W)
            
            return second_coords - sec_grid
        
        else:
            flat_disp = 1 / (fir_depths + self.eps)
            coord_homo = torch.cat([self.coord, torch.ones_like(flat_disp)], dim=2).view(-1, 4, H * W)

            flat_disp = flat_disp.view(-1, 1, H * W)
            rot_coord = torch.bmm(full_projection, coord_homo) 
            trans_disp = flat_disp * full_projection[:, :, 3:4].expand(-1, -1, H * W)
            tgt_coord = rot_coord + trans_disp
            
            tgt_coord = tgt_coord[:, :2, :] / (tgt_coord[:, 2:3, :] + self.eps)
            tgt_coord = tgt_coord.view(B, P, 2, H, W)
            return tgt_coord - self.coord[:, :, :2, :].view(B, P, 2, H, W)


class OccMask(torch.nn.Module):
    def __init__(self, th=3.0):
        super(OccMask, self).__init__()
        self.th = th
        self.base_coord = None

    def init_grid(self, shape, device):
        B, N, H, W = shape
        hh, ww = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float())
        coord = torch.zeros([B, N, H, W, 2])
        coord[..., 0] = ww
        coord[..., 1] = hh
        self.base_coord = coord.to(device)
        self.W = W
        self.H = H

    @torch.no_grad()
    def get_oob_mask(self, base_coord, flow_1_2):
        target_range = base_coord + flow_1_2.permute([0, 1, 3, 4, 2])
        oob_mask = (target_range[..., 0] < 0) | (target_range[..., 0] > self.W-1) | (
            target_range[..., 1] < 0) | (target_range[..., 1] > self.H-1)
        return ~oob_mask

    @torch.no_grad()
    def get_flow_inconsistency_tensor(self, base_coord, flow_1_2, flow_2_1):
        B, N, _, H, W = flow_1_2.shape
        sample_grids = base_coord + flow_1_2.permute([0, 1, 3, 4, 2])
        sample_grids[..., 0] /= (W - 1) / 2
        sample_grids[..., 1] /= (H - 1) / 2
        sample_grids -= 1
        sampled_flow = F.grid_sample(flow_2_1, sample_grids, align_corners=True)
        return torch.abs((sampled_flow + flow_1_2).sum(2, keepdim=True))

    def forward(self, flow_1_2, flow_2_1):
        """
            flow_1_2: [B, N, 2, H, W]
            flow_2_1: [B, N, 2, H, W]
        """
        B, N, _, H, W = flow_1_2.shape
        if self.base_coord is None:
            self.init_grid([B, N, H, W], device=flow_1_2.device)
        oob_mask = self.get_oob_mask(self.base_coord, flow_1_2)
        flow_inconsistency_tensor = self.get_flow_inconsistency_tensor(self.base_coord, flow_1_2, flow_2_1)
        valid_flow_mask = flow_inconsistency_tensor < self.th
        return valid_flow_mask * oob_mask


class FlowPredictor(nn.Module):

    def __init__(self,
                 sequence_length,
                 motion_mask_thre=0.6,
                 min_conf_thre=1.5,
                 eps=1e-6):

        super().__init__()
        self.sequence_length = sequence_length
        self.motion_mask_thre = motion_mask_thre
        self.min_conf_thre = min_conf_thre
        self.eps = eps
        self.depth_wrapper = DepthBasedWarping()
 
        self.flow_net = load_RAFT()
        self.flow_net.eval()
        self.sam_refiner = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    def get_flow(self, pair_imgs, iters=20):
        
        device = pair_imgs[0].device
        self.flow_net.to(device)
        
        with torch.no_grad():
            flow_fir_sec = self.flow_net(
                pair_imgs[0] * 255,
                pair_imgs[1] * 255,
                iters=iters,
                test_mode=True
            )[1]
            flow_sec_fir = self.flow_net(
                pair_imgs[1] * 255,
                pair_imgs[0] * 255,
                iters=iters,
                test_mode=True
            )[1]

        return flow_fir_sec, flow_sec_fir

    def get_motion_masks(self, batch, preds):
        
        N = len(batch)
        assert N == self.sequence_length
        pairs_idx = [
            (0, 1), (1, 2), (2, 3),
            (3, 4), (0, 3), (2, 4)
        ]
        device = preds[0]['pts3d'].device

        # RAFT flow
        flows_fir_sec = []
        flows_sec_fir = []
        for fir, sec in pairs_idx:
            pair_imgs = [
                (batch[fir]['img'] + 1.0) / 2.0, 
                (batch[sec]['img'] + 1.0) / 2.0
            ]
            flow_fir_sec, flow_sec_fir = self.get_flow(pair_imgs)
            flows_fir_sec.append(flow_fir_sec)
            flows_sec_fir.append(flow_sec_fir)
        flows_fir_sec = torch.stack(flows_fir_sec, dim=1)
        flows_sec_fir = torch.stack(flows_sec_fir, dim=1)

        flows_fir_sec.requires_grad_(False)
        flows_sec_fir.requires_grad_(False)

        # cam para
        B, H, W, _ = preds[0]['pts3d'].shape
        u, v = np.meshgrid(np.arange(W), np.arange(H))
        points_2d = np.stack((u, v), axis=-1)
        pp = torch.tensor((W/2, H/2))
        intrinsics = []
        extrinsics = []
        depths = []

        for idx in range(N):
            pts = preds[idx]['pts3d' if idx == 0 else 'pts3d_in_other_view'].detach().cpu()
            batch_extrinsics = []
            batch_intrinsics = []
            batch_depths = []
            batch_focals = []
            
            for batch_id in range(B):
                # use pts for pose estimation
                focal = estimate_focal_knowing_depth(pts[batch_id].unsqueeze(0), pp, focal_mode='weiszfeld')
                batch_focals.append(focal)
                
                if torch.all(focal > 0):
                    intrinsic = np.eye(4)
                    intrinsic[0, 0] = focal
                    intrinsic[1, 1] = focal
                    intrinsic[:2, 2] = pp
                    batch_intrinsics.append(intrinsic)

                    dist_coeffs = np.zeros(4).astype(np.float32)
                    success, rotation_vector, translation_vector, inliers = cv2.solvePnPRansac(
                        pts[batch_id].numpy().reshape(-1, 3).astype(np.float32), 
                        points_2d.reshape(-1, 2).astype(np.float32), 
                        intrinsic[:3, :3].astype(np.float32), 
                        dist_coeffs)
                    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                    extrinsic = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
                    extrinsic = np.vstack((extrinsic, [0, 0, 0, 1]))
                    batch_extrinsics.append(extrinsic)

                else:
                    intrinsic = np.eye(4)
                    intrinsic[:3, :3] = batch[idx]['camera_intrinsics'][batch_id].cpu().numpy()
                    extrinsic = batch[idx]['camera_pose'][batch_id].cpu().numpy()
                    batch_intrinsics.append(intrinsic)
                    batch_extrinsics.append(extrinsic)
                    print('Encount Bad Prediction')
              
            intrinsics.append(torch.tensor(np.stack(batch_intrinsics, axis=0)))
            extrinsics.append(torch.tensor(np.stack(batch_extrinsics, axis=0)))

            for batch_id in range(B):
                focal = batch_focals[batch_id]
                
                if torch.all(focal > 0):
                    
                    if idx == 0:
                        batch_depths.append(pts[batch_id][..., 2])
                    else:
                        rele_pose = inv(extrinsics[0][batch_id]) @ extrinsics[idx][batch_id]
                        batch_depths.append(geotrf(inv(rele_pose), pts[batch_id])[..., 2])

                else:
                    batch_depths.append(torch.tensor(batch[idx]['depthmap'][batch_id]))
            
            depths.append(torch.stack(batch_depths, dim=0))

        
        # get mask
        fir_intrinsics = []
        sec_intrinsics = []
        fir_extrinsics = []
        sec_extrinsics = []
        fir_depths = []
        sec_depths =[]

        for fir, sec in pairs_idx:
            fir_intrinsics.append(intrinsics[fir])
            sec_intrinsics.append(intrinsics[sec])
            fir_extrinsics.append(extrinsics[fir])
            sec_extrinsics.append(extrinsics[sec])
            fir_depths.append(depths[fir])
            sec_depths.append(depths[sec])
        
        fir_extrinsics = torch.stack(fir_extrinsics, dim=1).to(device=device, dtype=torch.float32)
        sec_extrinsics = torch.stack(sec_extrinsics, dim=1).to(device=device, dtype=torch.float32)
        fir_intrinsics = torch.stack(fir_intrinsics, dim=1).to(device=device, dtype=torch.float32)
        sec_intrinsics = torch.stack(sec_intrinsics, dim=1).to(device=device, dtype=torch.float32)
        fir_depths = torch.stack(fir_depths, dim=1).to(device=device, dtype=torch.float32)
        sec_depths = torch.stack(sec_depths, dim=1).to(device=device, dtype=torch.float32)

        ego_flow_fir_sec = self.depth_wrapper(
            fir_extrinsics, fir_intrinsics,
            sec_extrinsics, sec_intrinsics,
            fir_depths, use_depth=True
        )
        ego_flow_sec_fir = self.depth_wrapper(
            sec_extrinsics, sec_intrinsics,
            fir_extrinsics, fir_intrinsics,
            sec_depths, use_depth=True
        )

        err_map_fir = torch.norm(ego_flow_fir_sec[:, :, :2, ...] - flows_fir_sec, dim=2)
        err_map_sec = torch.norm(ego_flow_sec_fir[:, :, :2, ...] - flows_sec_fir, dim=2)
        err_map_fir = (
            (err_map_fir - err_map_fir.amin(dim=(2, 3), keepdim=True)) /
            (err_map_fir.amax(dim=(2, 3), keepdim=True) - err_map_fir.amin(dim=(2, 3), keepdim=True))
        )
        err_map_sec = (
            (err_map_sec - err_map_sec.amin(dim=(2, 3), keepdim=True)) /
            (err_map_sec.amax(dim=(2, 3), keepdim=True) - err_map_sec.amin(dim=(2, 3), keepdim=True))
        )

        dynamic_masks = [[] for _ in range(self.sequence_length)]
        for pair_id in range(len(pairs_idx)):
            fir, sec = pairs_idx[pair_id]
            dynamic_masks[fir].append(err_map_fir[:, pair_id, ...])
            dynamic_masks[sec].append(err_map_sec[:, pair_id, ...])
        
        for frame_id in range(self.sequence_length):
            dynamic_masks[frame_id] = torch.stack(dynamic_masks[frame_id], dim=1).mean(dim=1)

        return dynamic_masks
        
    def sam2_refiner(self, 
                     imgs, 
                     dynamic_masks,
                     ann_obj_id=1
                    ):

        for frame_id in range(self.sequence_length): 
            dynamic_masks[frame_id] = dynamic_masks[frame_id] > self.motion_mask_thre
        sam_masks = [[] for _ in range(self.sequence_length)]
        
        imgs = (imgs + 1.0) / 2.0
        B, N, _, H, W = imgs.shape
        device = imgs.device
        self.sam_refiner.to(device)
        autocast_dtype = torch.bfloat16

        with torch.autocast(device_type='cuda', dtype=autocast_dtype):
            for batch in range(B):
                batch_imgs = imgs[batch]
                inference_state = self.sam_refiner.init_state(video_path=batch_imgs)
                batch_masks = [dynamic_masks[frame_id][batch, ...] for frame_id in range(self.sequence_length)]
                self.sam_refiner.reset_state(inference_state)
                for frame_id, mask in enumerate(batch_masks):
                    _, out_obj_ids, out_mask_logits = self.sam_refiner.add_new_mask(
                        inference_state,
                        frame_idx=frame_id,
                        obj_id=ann_obj_id,
                        mask=mask
                    )
                video_segments = {}
                for out_frame_idx, out_obj_ids, out_mask_logits in self.sam_refiner.propagate_in_video(inference_state, start_frame_idx=0):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
                for out_frame_idx in range(self.sequence_length):
                    sam_masks[out_frame_idx].append(video_segments[out_frame_idx][ann_obj_id])
        
        # update mask
        sam_masks = [
            np.stack(frame_sam_masks, axis=0).squeeze(1)
            for frame_sam_masks in sam_masks
        ]
        refined_masks = [[] for _ in range(self.sequence_length)]
        for frame_id in range(self.sequence_length):
            frame_sam_masks = torch.from_numpy(sam_masks[frame_id]).to(device)
            frame_dym_masks = dynamic_masks[frame_id].to(device)
            refined_masks[frame_id] = frame_sam_masks | frame_dym_masks
        
        return refined_masks
         
    def forward(self, batch, preds):

        dynamic_masks = self.get_motion_masks(batch, preds)

        imgs = [batch_views['img'] for batch_views in batch]
        imgs = torch.stack(imgs, dim=1)
        refined_dynamic_masks = self.sam2_refiner(imgs, dynamic_masks)

        return refined_dynamic_masks