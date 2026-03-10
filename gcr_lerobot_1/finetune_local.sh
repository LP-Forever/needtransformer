deepspeed lerobot/scripts/dps_train.py \
    --deepspeed="./ds_zero2.json" \
    --policy.type="pi0" \
    --dataset.root="/Data/lerobot_data/real_world" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix="simpler_bridge" \
    --wandb.enable=true \
    --wandb.project="pi0-ft-simulated" \
    --job_name="pi0-04-21-ft-vlabench-local-bs-128-cos-sche" \
    --log_dir="logs" \
    --output_dir="/Data/lzl/pi0-ft-simulated/0421-ft-vlabench-bs-128-1st-cos-sche" \
    --steps=30_000 \
    # --dataset.image_transforms.enable=true