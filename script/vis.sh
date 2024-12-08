python align_and_vis.py \
    --sequence_length 5 \
    --dataset "NuSceneDataset(
        split='val',
        data_root='datasets/nuscenes',
        sequence_length=5,
        cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        resolution=224,
        depth_root='third_party/r3d3/pred/samples'
    )" \
    --resolution 224 \
    --save_path ./output/nuscenes/ \
    --ckpt_path ./checkpoints/driv3r.pth