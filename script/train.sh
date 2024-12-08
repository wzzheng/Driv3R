torchrun \
  --nproc_per_node=8 \
  train.py \
  --train_dataset "NuSceneDataset(
    split='train',
    data_root='datasets/nuscenes',
    sequence_length=5,
    cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
    resolution=224,
    depth_root='third_party/r3d3/pred/samples'
  )" \
  --test_dataset "NuSceneDataset(
    split='val',
    data_root='datasets/nuscenes',
    sequence_length=5,
    cams=['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
    resolution=224,
    depth_root='third_party/r3d3/pred/samples'
  )" \
  --pretrained checkpoints/spann3r.pth \
  --batch_size 4