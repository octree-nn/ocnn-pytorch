# train - classification
python classification.py  \
       --config configs/cls_m40.yaml  \
       SOLVER.alias ref   \
       SOLVER.gpu 0,

python classification.py  \
       --config configs/cls_m40.yaml  \
       MODEL.name resnet  \
       SOLVER.alias ref_resnet   \
       SOLVER.gpu 1,

# test - classification
python classification.py  \
       --config configs/cls_m40.yaml  \
       SOLVER.run test  \
       SOLVER.ckpt logs/m40/m40_03231708/checkpoints/00300.model.pth


# eval - scannet segmentation
alias=r4
python segmentation.py   \
       --config configs/seg_scannet_eval.yaml  \
       SOLVER.eval_epoch 12  \
       SOLVER.alias ${alias} \
       SOLVER.ckpt logs/scannet/D9_2cm/checkpoints/00600.model.pth  \
       DATA.test.location  data/scannet/train  \
       DATA.test.filelist data/scannet/scannetv2_val_new.txt

python tools/seg_scannet.py  \
       --run generate_output_seg   \
       --path_in data/scannet/train  \
       --path_pred logs/scannet/D9_2cm_eval_${alias}  \
       --path_out logs/scannet/D9_2cm_eval_seg_${alias}  \
       --filelist  data/scannet/scannetv2_val_new.txt

python tools/seg_scannet.py  \
       --run calc_iou   \
       --path_in data/scannet/train \
       --path_pred logs/scannet/D9_2cm_eval_seg_${alias}


# def octree2voxel(data, octree):
#   r''' Consistent with v1.
#   '''
#   batch_size = octree.batch_size
#   vox = data.view(batch_size, -1, data.shape[1])  # (B, H, C)
#   vox = vox.permute(0, 2, 1)  # (B, H, C) -> (B, C, H)
#   return vox
