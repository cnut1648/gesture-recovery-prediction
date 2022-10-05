source activate meta;

k=2
ckpt="/home/jiashu/seq/selected_ckpt/3681/epoch09-AUROC0.74-acc0.50.ckpt"
id="3681"

# 1. vis attn scores, generate images & scores in artifact/IMV/id
# python IMV_vis.py --k $k --ckpt_path $ckpt --ckpt_id $id

# 2. attn occl from attn scores in artifact/attn_occl/id
# python attn_occlusion.py --k $k --ckpt_path $ckpt --ckpt_id $id

# 3. vis coarse attn occlusion scores in artifact/attn_occl/id-coarse
python attn_occlusion_vis.py --k $k --ckpt_path $ckpt --ckpt_id $id

python attn_occlusion_vis.py --k $k --ckpt_path $ckpt --ckpt_id $id --coarse