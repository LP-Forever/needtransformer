#conda activate lerobot_v2_1
#export MASTER_PORT=$(shuf -i 20000-60000 -n 1)
export PYTHONPATH=$PYTHONPATH:$(pwd)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    #python -m deepspeed.launcher.launch --num_gpus=8 --master_port=29500 ./lerobot/scripts/dps_train_2.py \
python -m lerobot.scripts.dps_train_2 \
    --deepspeed="./ds_zero2_40G.json" \
    --policy.type="pi0" \
    --policy.use_lora=true \
    --policy.loss_type="mse_loss" \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=true \
    --policy.add_new_tokens=false \
    --dataset.root="/mnt/wangxiaofa/robot_dataset/lerobot-format" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix="sauce_hz_4" \
    --dataset.image_transforms.enable=false \
    --wandb.enable=true \
    --wandb.project="NEW-pi05" \
    --job_name="0428-pi05-sauce-submitat1804" \
    --log_dir="/mnt/wangxiaofa/logs" \
    --output_dir="/mnt/wangxiaofa/amlt_output/sauce-0428-1804" \
    --steps=60000 \
    --save_freq=10000 \
    --log_freq=40 \
    --policy.pt_weight_path="/mnt/wangxiaofa/pi0_pretrain/model_new.pt"
    # --policy.pt_weight_path="/scratch/amlt_code/model_new.pt" \
    # --policy.pt_weight_path="/Data/lzl/weights/pi_zero_pt/model.pt" \
    # --policy.pretrained_path=""/Data/lzl/pi0-ft-real/1124-latent-pi05-ft-real/step30000/mp_rank_00_model_states.pt"
    # --dataset.image_transforms.enable=true
    # "/Data/lzl/openpi/pytorch/pi05_base/model_new.pt"