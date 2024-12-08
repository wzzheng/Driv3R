import os
import cv2
import tqdm
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from PIL import Image
import torchvision

from torch.utils.data import DataLoader

from dust3r.losses import L21
from driv3r.datasets import *
from driv3r.model import Spann3R
from driv3r.loss import Regr3D_t_ScaleShiftInv

def get_args_parser():
    parser = argparse.ArgumentParser('Driv3R Aligner and Visualization', add_help=False)
    parser.add_argument('--dataset', required=True, type=str, help="vis dataset")
    parser.add_argument('--sequence_length', required=True, type=int, help="sequence length")
    parser.add_argument('--resolution', required=True, type=int, help="resolution")
    parser.add_argument('--save_path', type=str, default='./output/nuscenes/', help='Path to experiment folder')
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/driv3r.pth', help='ckpt path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--scale', type=int, default=120, help="point scale")
    parser.add_argument('--conf_thresh', type=float, default=1.001, help='confidence threshold')

    return parser

@torch.no_grad()
def main(args):

    per_frame = False
    sequence_length = args.sequence_length
    save_dir = args.save_path
    scale = args.scale
    H = args.resolution
    W = args.resolution
    os.makedirs(save_dir, exist_ok=True)

    # Load model
    model = Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', 
                use_feat=False).to(args.device) 
    model.load_state_dict(torch.load(args.ckpt_path)['model'])
    model.eval()

    # dataloader
    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    dataloader = build_dataset(args.dataset, batch_size=1, num_workers=0, test=True)

    batch_outputs = {}
    for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):

        scene_id = batch_id // len(cams) + 1
        os.makedirs(os.path.join(save_dir, f'scene_{str(scene_id)}'), exist_ok=True)
        
        ##### Inference
        for view in batch:
            view['img'] = view['img'].to(args.device, non_blocking=True)
        
        left_batch = batch[0::2]
        right_batch = batch[1::2]

        left_preds, left_preds_all = model.forward(left_batch)
        right_preds, right_preds_all = model.forward(right_batch)
        cam = cams[batch_id % len(cams)]
        batch_outputs[cam] = {
            'left': {
                'inputs': left_batch,
                'preds': left_preds,
                'preds_all': left_preds_all
            },
            'right': {
                'inputs': right_batch,
                'preds': right_preds,
                'preds_all': right_preds_all
            }
        }

     
        splits = ['left', 'right']

        # save complete scene
        if (batch_id + 1) % len(cams) == 0:

            if per_frame:
                frame_pcd = {idx: {
                    'image': [],
                    'pts': [],
                    'conf': []
                } for idx in range(sequence_length)}

            all_points = []
            all_colors = []
            all_confs = []

            for cam in cams:

                for split in splits:
                    
                    preds = batch_outputs[cam][split]['preds']
                    inputs = batch_outputs[cam][split]['inputs']
                    preds_all = batch_outputs[cam][split]['preds_all']

                    # load lidar point
                    batch = NuSceneDataset.load_lidar_pts(inputs)
                    for view in batch:
                        for item in view:
                            if isinstance(view[item], torch.Tensor):
                                view[item] = view[item].to(args.device, non_blocking=True)

                    gt_pts, pred_pts, gt_factor, pr_factor, masks, monitoring = criterion.get_all_pts3d_t(batch, preds_all)
                    pred_scale, gt_scale, pred_shift_z, gt_shift_z  = monitoring['pred_scale'], monitoring['gt_scale'], monitoring['pred_shift_z'], monitoring['gt_shift_z']

                    in_camera1 = None
                    pts_all = []
                    pts_gt_all = []
                    images_all = []
                    masks_all = []
                    conf_all = []

                    for j, view in enumerate(batch):

                        if in_camera1 is None:
                            in_camera1 = view['camera_pose'][0].cpu()

                        # save image
                        img = view['img'].cpu()[0]
                        img = (img + 1.0) / 2.0
                        
                        image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
                        mask = view['valid_mask'].cpu().numpy()[0]
                        pts = pred_pts[0][j].cpu().numpy()[0] if j < len(pred_pts[0]) else pred_pts[1][-1].cpu().numpy()[0]
                        conf = preds[j]['conf'][0].cpu().data.numpy()
                        K = view['camera_intrinsics'][0].cpu().numpy()
                        extrinsic = view['camera_pose'][0].cpu().numpy()
                        
                        depth_pred = pts[..., -1]
                        depth_pred = (depth_pred - np.min(depth_pred)) / (np.max(depth_pred) - np.min(depth_pred))

                        pts_lidar = view['pts3d'][0].cpu().numpy()[mask]
                        N = pts_lidar.shape[0]
                        pts_lidar_homo = np.concatenate([pts_lidar, np.ones((N, 1), dtype=np.float32)], axis=-1)
                        pts_gt_c = (np.linalg.inv(extrinsic) @ pts_lidar_homo.T).T
                        shift = np.min(pts_gt_c[..., 2])

                        depth_pred = depth_pred * scale + shift
                        depth_pred = depth_pred.reshape(-1)[:, np.newaxis]
                        image = ((image + 1.0) / 2.0).reshape(-1, 3)

                        u, v = np.meshgrid(np.arange(W), np.arange(H))
                        pixel_coords = np.stack([u, v, np.ones_like(u)], axis=-1)

                        K_inv = np.linalg.inv(K)
                        depth_values = depth_pred.reshape(-1)  # (H * W,)
                        pixel_coords_3d = pixel_coords.reshape(-1, 3)  # (H * W, 3)
                        camera_coords = (K_inv @ pixel_coords_3d.T).T * depth_values[:, np.newaxis]  # (H * W, 3)
                        extrinsic_rotation = extrinsic[:3, :3]  # (3, 3)
                        extrinsic_translation = extrinsic[:3, 3]  # (3,)
                        world_coords = camera_coords @ extrinsic_rotation.T + extrinsic_translation
                        
                        images_all.append(image[None, ...])
                        pts_all.append(world_coords[None, ...])
                        masks_all.append(mask[None, ...])
                        conf_all.append(conf[None, ...])

                        if per_frame:
                            frame_pcd[j]['image'].append(image[None, ...])
                            frame_pcd[j]['pts'].append(world_coords[None, ...])
                            frame_pcd[j]['conf'].append(conf[None, ...])
                    
                    images_all = np.concatenate(images_all, axis=0)
                    pts_all = np.concatenate(pts_all, axis=0)
                    masks_all = np.concatenate(masks_all, axis=0)
                    conf_all = np.concatenate(conf_all, axis=0)

                    all_points.append(pts_all.reshape(-1, 3))
                    all_colors.append(images_all.reshape(-1, 3))
                    all_confs.append(conf_all.reshape(-1))
            
            all_points = np.concatenate(all_points, axis=0)
            all_colors = np.concatenate(all_colors, axis=0)
            all_confs = np.concatenate(all_confs, axis=0)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(all_points)
            pcd.colors = o3d.utility.Vector3dVector(all_colors)
            o3d.io.write_point_cloud(os.path.join(save_dir, f'scene_{str(scene_id)}', 'pcd.ply'), pcd)

            if per_frame:
                for j in range(sequence_length):
                    frame_pcd[j]['image'] = np.concatenate(frame_pcd[j]['image'], axis=0).reshape(-1, 3)
                    frame_pcd[j]['pts'] = np.concatenate(frame_pcd[j]['pts'], axis=0).reshape(-1, 3)
                    frame_pcd[j]['conf'] = np.concatenate(frame_pcd[j]['conf'], axis=0).reshape(-1)

                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(frame_pcd[j]['pts'])
                    pcd.colors = o3d.utility.Vector3dVector(frame_pcd[j]['image'])
                    o3d.io.write_point_cloud(os.path.join(save_dir, f'scene_{str(scene_id)}', f'pcd_frame_{j}.ply'), pcd)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)