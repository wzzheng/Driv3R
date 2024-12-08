import os
import cv2
import torch
import argparse
import numpy as np
import open3d as o3d
import os.path as osp
from PIL import Image
import torchvision
import tqdm, json
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from dust3r.utils.geometry import geotrf
from dust3r.losses import L21
from driv3r.datasets import *
from driv3r.model import Spann3R
from driv3r.loss import Regr3D_t_ScaleShiftInv
from driv3r.tools.eval_recon import accuracy, completion

def get_args_parser():
    parser = argparse.ArgumentParser('Driv3R Evaluation', add_help=False)
    parser.add_argument('--ckpt_path', type=str, default='./checkpoints/driv3r.pth', help='ckpt path')
    parser.add_argument('--dataset', required=True, type=str, help="eval dataset")
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--resolution', required=True, type=int, help="resolution")
    parser.add_argument('--sequence_length', required=True, type=int, help="sequence_length")
    return parser

@torch.no_grad()
def main(args):

    H = args.resolution
    W = args.resolution
    sequence_length = args.sequence_length
    threshold = 100
        
    # Load model
    model = Spann3R(dus3r_name='./checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', 
                use_feat=False).to(args.device)
    
    model.load_state_dict(torch.load(args.ckpt_path)['model'])
    model.eval()

    criterion = Regr3D_t_ScaleShiftInv(L21, norm_mode=False, gt_scale=True)
    dataloader = build_dataset(args.dataset, batch_size=1, num_workers=0, test=True)

    acc_all = 0
    acc_all_med = 0
    comp_all = 0
    comp_all_med = 0
    nc1_all = 0
    nc1_all_med = 0
    nc2_all = 0
    nc2_all_med = 0
    abs_rel_all = 0
    sq_rel_all = 0
    rmse_all = 0
    delta_1_25_all = 0
    delta_1_25_2_all = 0
    total_sequences = 0

    for batch_id, batch in tqdm.tqdm(enumerate(dataloader)):
        
        batch_outputs = {}
        scene_id = batch_id // len(cams) + 1
        cam = cams[batch_id % len(cams)]   

        ##### Inference
        for view in batch:
            view['img'] = view['img'].to(args.device, non_blocking=True)
        
        if len(batch) == 2 * sequence_length:
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
            total_sequences += 2
            splits = ['left', 'right']

        elif len(batch) == sequence_length:
            preds, preds_all = model.forward(batch)
            cam = cams[batch_id % len(cams)]
            batch_outputs[cam] = {
                'single': {
                    'inputs': batch,
                    'preds': preds,
                    'preds_all': preds_all
                }
            }
            total_sequences += 1
            splits = ['single']

        else:
            raise NotImplementedError

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
            abs_rel = 0
            sq_rel = 0
            rmse = 0
            delta_1_25 = 0
            delta_1_25_2 = 0

            for j, view in enumerate(batch):
                if in_camera1 is None:
                    in_camera1 = view['camera_pose'][0].cpu()
        
                image = view['img'].permute(0, 2, 3, 1).cpu().numpy()[0]
                mask = view['valid_mask'].cpu().numpy()[0]
                pts = pred_pts[0][j].cpu().numpy()[0] if j < len(pred_pts[0]) else pred_pts[1][-1].cpu().numpy()[0]
                conf = preds[j]['conf'][0].cpu().data.numpy()

                pts_gt = gt_pts[j].detach().cpu().numpy()[0]

                #### Align predicted 3D points to the ground truth 
                pts[..., -1] += gt_shift_z.cpu().numpy().item()
                depth_pred = pts[..., -1][mask]             
                pts = geotrf(in_camera1, pts)
                
                pts_gt[..., -1] += gt_shift_z.cpu().numpy().item()
                depth_gt = pts_gt[..., -1][mask]
                pts_gt = geotrf(in_camera1, pts_gt)

                abs_rel += np.mean(np.abs(depth_pred - depth_gt) / depth_gt)
                sq_rel += np.mean(((depth_pred - depth_gt) / depth_gt) ** 2)
                rmse += np.sqrt(np.mean((depth_pred - depth_gt) ** 2))
                delta_1_25 += np.mean(np.maximum(depth_pred / depth_gt, depth_gt / depth_pred) < 1.25)
                delta_1_25_2 += np.mean(np.maximum(depth_pred / depth_gt, depth_gt / depth_pred) < 1.5625)         

                images_all.append((image[None, ...] + 1.0)/2.0)
                pts_all.append(pts[None, ...])
                pts_gt_all.append(pts_gt[None, ...])
                masks_all.append(mask[None, ...])
                conf_all.append(conf[None, ...])
            
            abs_rel /= len(batch)
            sq_rel /= len(batch)
            rmse /= len(batch)
            delta_1_25 /= len(batch)
            delta_1_25_2 /= len(batch)
            
            images_all = np.concatenate(images_all, axis=0)
            pts_all = np.concatenate(pts_all, axis=0)
            pts_gt_all = np.concatenate(pts_gt_all, axis=0)
            masks_all = np.concatenate(masks_all, axis=0)
            conf_all = np.concatenate(conf_all, axis=0)
            
            # accuracy, completion, normal consistent
            pts_all_masked = pts_all
            images_all_masked = images_all
            pts_gt_all_masked = pts_gt_all[masks_all > 0]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pts_all_masked.reshape(-1, 3))
            pcd.colors = o3d.utility.Vector3dVector(images_all_masked.reshape(-1, 3))
    
            pcd_gt = o3d.geometry.PointCloud()
            pcd_gt.points = o3d.utility.Vector3dVector(pts_gt_all_masked.reshape(-1, 3))

            trans_init = np.eye(4)
            reg_p2p = o3d.pipelines.registration.registration_icp(
                pcd, pcd_gt, threshold, trans_init,
                o3d.pipelines.registration.TransformationEstimationPointToPoint())
            
            transformation = reg_p2p.transformation
                        
            pcd = pcd.transform(transformation)
            pcd.estimate_normals()
            pcd_gt.estimate_normals()

            gt_normal = np.asarray(pcd_gt.normals)
            pred_normal = np.asarray(pcd.normals)

            acc, acc_med, nc1, nc1_med = accuracy(pcd_gt.points, pcd.points, gt_normal, pred_normal)
            comp, comp_med, nc2, nc2_med = completion(pcd_gt.points, pcd.points, gt_normal, pred_normal)

            acc_all += acc
            comp_all += comp
            nc1_all += nc1
            nc2_all += nc2

            acc_all_med += acc_med
            comp_all_med += comp_med
            nc1_all_med += nc1_med
            nc2_all_med += nc2_med

            abs_rel_all += abs_rel
            sq_rel_all += sq_rel
            rmse_all += rmse
            delta_1_25_all += delta_1_25
            delta_1_25_2_all += delta_1_25_2

    print('accuracy', acc_all / total_sequences)
    print('completion', comp_all / total_sequences)
    print('nc1', nc1_all / total_sequences)
    print('nc2', nc2_all / total_sequences)
    print('acc_med', acc_all_med / total_sequences)
    print('comp_med', comp_all_med / total_sequences)
    print('nc1_med', nc1_all_med / total_sequences)
    print('nc2_med', nc2_all_med / total_sequences)
    print('abs_rel', abs_rel_all / total_sequences)
    print('sq_rel', sq_rel_all / total_sequences)
    print('rmse', rmse_all / total_sequences)
    print('delta_1_25', delta_1_25_all / total_sequences)
    print('delta_1_25_2', delta_1_25_2_all / total_sequences)
        
if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)