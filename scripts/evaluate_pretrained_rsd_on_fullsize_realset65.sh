lr_path="data/validation/RealSet65"
sr_path="data/validation/RealSet65_sr_results"
ckpt_path="logs/pretrained_rsd/ema_model_2800.pth"
dataset_name="realset65"
metrics_fold_path="metrics_fold"
# CUDA_VISIBLE_DEVICES=0 python inference.py --task RSD --config ./configs/RSD.yaml --one_step --ckpt ${ckpt_path} --chop_size 512 --in_path ${lr_path} --out_path ${sr_path} --scale 4
python evaluate.py --dataset ${dataset_name} -bs 1 --inference_fold ${sr_path} --gt_fold "" --metrics_fold ${metrics_fold_path}