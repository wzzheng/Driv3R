python eval.py \
    --dataset "NuSceneDataset(
        split='val',
        data_root='datasets/nuscenes',
        sequence_length=5,
        cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
        resolution=224,
        depth_root='third_party/r3d3/pred/samples',
        dynamic=True,
        dynamic_metas='preprocess/dynamic_metas.pkl'
    )" \
    --ckpt_path ./checkpoints/driv3r.pth \
    --resolution 224 \
    --sequence_length 5