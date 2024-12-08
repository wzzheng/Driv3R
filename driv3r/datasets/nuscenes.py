import os, json, pickle, tqdm
import torch, cv2
from torch.utils.data import Dataset
from pyquaternion import Quaternion
import numpy as np
import open3d as o3d

from dust3r.utils.image import imread_cv2
from driv3r.datasets.base_many_view_dataset import BaseManyViewDataset

class NuSceneDataset(BaseManyViewDataset):

    def __init__(self, 
                 data_root,
                 sequence_length,
                 cams, 
                 depth_root,
                 dynamic=False,
                 dynamic_metas=None,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)
        self.data_root = data_root
        self.sequence_length = sequence_length
        self.cams = cams
        self.depth_root = depth_root
        self.dynamic = dynamic
        self.dynamic_metas = dynamic_metas
        self.dynamic_scene_list = []

        if self.split == 'train':
            with open(os.path.join(self.data_root, "train_metas.pkl"), 'rb') as f:
                self.metas = pickle.load(f)
        elif self.split == 'val':
            with open(os.path.join(self.data_root, "val_metas.pkl"), 'rb') as f:
                self.metas = pickle.load(f)
        else:
            raise NotImplementedError
        
        self.scene_list = self.load_scenes()

        if self.dynamic:
            with open(self.dynamic_metas, 'rb') as f:
                self.dynamic_meta_list = pickle.load(f)
                sequence_ids = list(self.dynamic_meta_list.keys())

                for sequence_id in sequence_ids:
                    sequence_infos = self.scene_list[sequence_id]
                    dynamic_sequence_infos = []
                    for sequence_info in sequence_infos:
                        sequence_info['splits'] = self.dynamic_meta_list[sequence_id]
                        dynamic_sequence_infos.append(sequence_info)
                    self.dynamic_scene_list.append(dynamic_sequence_infos)

    def load_scenes(self):
        
        sequence_length = self.sequence_length
        scene_list = []

        for scene_token, scene_metas in self.metas.items():
            # sort by timestamp
            scene_metas = dict(sorted(scene_metas.items(), key=lambda item: item[1]["CAM_FRONT"]["timestamp"]))
            sample_token_list = [sample_token for sample_token in scene_metas.keys()]

            for video_idx in range(len(scene_metas) // sequence_length):
                # sample token for current sequence
                sample_tokens = sample_token_list[video_idx * sequence_length : (video_idx + 1) * sequence_length]
                for cam in self.cams:
                    video_metas = []
                    for token in sample_tokens:
                        video_metas.append(dict(
                            timestamp=scene_metas[token][cam]["timestamp"],
                            img=os.path.join(self.data_root, scene_metas[token][cam]["filename"]),
                            pcd=os.path.join(self.data_root, scene_metas[token]["LIDAR_TOP"]["filename"]),
                            ego_pose=scene_metas[token][cam]["ego_pose"],
                            lidar_pose=scene_metas[token]["LIDAR_TOP"]["calibrated_sensor"],
                            cam_poses=scene_metas[token][cam]["calibrated_sensor"],
                            image_wh=scene_metas[token][cam]["image_wh"]
                        ))
                    # input metas
                    sequence_metas = []
                    for sample_idx in range(sequence_length):

                        cur_sample_meta = video_metas[sample_idx]
                        cam_poses = cur_sample_meta["cam_poses"]
                        e2w = cur_sample_meta["ego_pose"]
                        l2e = cur_sample_meta["lidar_pose"]
                        c2e_matrixs = NuSceneDataset.get_extrinsic_matrix(
                            rot=cam_poses["rotation"],
                            trans=cam_poses["translation"]
                        )
                        l2e_matrix = NuSceneDataset.get_extrinsic_matrix(
                            rot=l2e["rotation"],
                            trans=l2e["translation"]
                        )
                        e2w_matrix = NuSceneDataset.get_extrinsic_matrix(
                            rot=e2w["rotation"],
                            trans=e2w["translation"]
                        )

                        sequence_metas.append(dict(
                            timestamp=cur_sample_meta["timestamp"],
                            img=cur_sample_meta["img"],
                            pcd=cur_sample_meta["pcd"],
                            camera_intrinsics=np.array(cam_poses["intrinsics"]),
                            lidar_pose=e2w_matrix @ l2e_matrix,
                            cam_poses=e2w_matrix @ c2e_matrixs,
                            true_shape=cur_sample_meta["image_wh"]
                        ))

                    scene_list.append(sequence_metas)
            
        return scene_list

    def __len__(self):
        if self.dynamic:
            return len(self.dynamic_scene_list)
        else:
            return len(self.scene_list)

    @staticmethod
    def get_extrinsic_matrix(rot, trans):
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = Quaternion(rot).rotation_matrix
        extrinsic[:3, 3] = np.array(trans)
        return extrinsic

    @staticmethod
    def load_lidar_pts(views, image_size=(224, 224)):

        h, w = image_size
        device = views[0]['img'].device

        new_views = []
        # Load Lidar points for evaluation, assert batch_size = 1
        for view in views:

            pts3d = torch.zeros((1, h, w, 3), dtype=torch.float32, device=device)
            valid_mask = torch.zeros((1, h, w), dtype=torch.bool, device=device)

            pcd_l = np.fromfile(view["pcd"][0], dtype=np.float32)
            pcd_l = pcd_l.reshape(-1, 5)[:, :3]
            N = pcd_l.shape[0]
            ones = np.ones((N, 1), dtype=np.float32)
            pcd_l_homo = np.concatenate((pcd_l, ones), axis=-1)

            l2w_matrix = view["lidar_pose"][0].numpy()
            c2w_matrix = view["camera_pose"][0].numpy()
            l2c_matrix = np.linalg.inv(c2w_matrix) @ l2w_matrix

            pcd_c_homo = (l2c_matrix @ pcd_l_homo.T).T
            pcd_w_homo = (l2w_matrix @ pcd_l_homo.T).T 
            z_mask = pcd_c_homo[:, 2] > 0

            K = np.eye(4, 4, dtype=np.float32)
            K[:3, :3] = view['camera_intrinsics']
            pcd_pixel_homo = (K @ pcd_c_homo.T).T
            pcd_pixel_homo[:, 0] /= (pcd_pixel_homo[:, 2] + 1e-7)
            pcd_pixel_homo[:, 1] /= (pcd_pixel_homo[:, 2] + 1e-7)
            pcd_pixel = pcd_pixel_homo[:, :2].astype(np.int32)
            xy_mask = (pcd_pixel[:, 0] >= 0) & (pcd_pixel[:, 0] < w) & (pcd_pixel[:, 1] >= 0) & (pcd_pixel[:, 1] < h)
            
            valid_point_mask = xy_mask & z_mask
            valid_lidar_pts = pcd_w_homo[valid_point_mask][:, :3]
            valid_pcd_pixel = torch.tensor(pcd_pixel[valid_point_mask], dtype=torch.long, device=device)

            y_indices = valid_pcd_pixel[:, 1]
            x_indices = valid_pcd_pixel[:, 0]
            pts3d[:, y_indices, x_indices, :] = torch.tensor(valid_lidar_pts, dtype=torch.float32, device=device)
            valid_mask[:, y_indices, x_indices] = True

            view['pts3d'] = pts3d
            view['valid_mask'] = valid_mask
            new_views.append(view)
        
        return new_views

    def get_depth_map_via_lidar(self, sample_meta, image_size):
        
        h, w, _ = image_size
        pcd_l = np.fromfile(sample_meta["pcd"], dtype=np.float32)
        pcd_l = pcd_l.reshape(-1, 5)[:, :3]
        N = pcd_l.shape[0]
        ones = np.ones((N, 1), dtype=np.float32)
        pcd_l_homo = np.concatenate((pcd_l, ones), axis=-1)

        l2w_matrix = sample_meta["lidar_pose"]
        c2w_matrix = sample_meta["camera_pose"]
        l2c_matrix = np.linalg.inv(c2w_matrix) @ l2w_matrix

        pcd_c_homo = (l2c_matrix @ pcd_l_homo.T).T

        K = np.eye(4, 4, dtype=np.float32)
        K[:3, :3] = sample_meta['camera_intrinsics']
        pcd_pixel_homo = (K @ pcd_c_homo.T).T
        pcd_pixel_homo[:, 0] /= (pcd_pixel_homo[:, 2] + 1e-7)
        pcd_pixel_homo[:, 1] /= (pcd_pixel_homo[:, 2] + 1e-7)
        pcd_pixel = pcd_pixel_homo[:, :2].astype(np.int32)
        x, y = pcd_pixel.T

        depth_map = np.zeros((h, w), dtype=np.float32)
        depth_map[y.clip(min=0, max=h-1), x.clip(min=0, max=w-1)] = pcd_c_homo[:, 2]

        return depth_map
    
    def get_r3d3_depth_map(self, 
                           meta,
                           true_image_shape, 
                           resolution,
                           nus_intri):
        
        h_nus, w_nus, _ = true_image_shape
        rgb_filename = os.path.basename(meta["img"])
        cam = rgb_filename.split("__")[1]
        r3d3_depth_path = os.path.join(
            self.depth_root, cam,
            rgb_filename.replace('.jpg', '_depth(0)_pred.npz')
        )
        assert os.path.exists(r3d3_depth_path)
        
        r3d3_npz = np.load(r3d3_depth_path)
        r3d3_depth = r3d3_npz['depth']
        
        r3d3_depth = np.pad(r3d3_depth, ((1, 1), (16, 16)), mode='constant', constant_values=0)
        r3d3_depth = cv2.resize(r3d3_depth, (w_nus, h_nus), interpolation=cv2.INTER_LANCZOS4)
   
        return r3d3_depth
        
    def _get_views(self, idx, resolution, rng):
        
        views = []
        if self.dynamic:
            video_frames = self.dynamic_scene_list[idx]
        else:
            video_frames = self.scene_list[idx]
    

        for frame_meta in video_frames:
            rgb_image = imread_cv2(frame_meta["img"])
            intrinsics = frame_meta['camera_intrinsics'].astype(np.float32)
            depthmap = self.get_r3d3_depth_map(
                meta=frame_meta, 
                true_image_shape=rgb_image.shape, 
                resolution=resolution,
                nus_intri=intrinsics
            )
            
            # split into two frames, left and right
            if self.dynamic:
                splits = frame_meta['splits']
            else:
                splits = ['left', 'right']
            for split in splits:
                split_rgb_image, split_depthmap, split_intrinsics = self._crop_resize_if_necessary(
                    rgb_image, depthmap, intrinsics, resolution, 
                    rng=rng, info=frame_meta["img"], split=split
                )
            
                views.append(dict(
                    img=split_rgb_image,
                    depthmap=split_depthmap,
                    camera_pose=frame_meta['cam_poses'].astype(np.float32),
                    pcd=frame_meta['pcd'],
                    camera_intrinsics=split_intrinsics,
                    lidar_pose=frame_meta['lidar_pose'].astype(np.float32),
                    dataset='nuscenes',
                    label=frame_meta["img"],
                    instance=os.path.basename(frame_meta["img"])
                ))
        
        return views


if __name__ == '__main__':

    cams=[
        'CAM_FRONT',
        'CAM_FRONT_LEFT',
        'CAM_FRONT_RIGHT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT'
    ]

    dataset = NuSceneDataset(
        data_root='datasets/nuscenes',
        sequence_length=5,
        cams=cams,
        split='val',
        depth_root='third_party/r3d3/pred/samples',
        resolution=224
    )