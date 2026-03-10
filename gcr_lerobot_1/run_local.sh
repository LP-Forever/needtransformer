#conda activate lerobot_v2_1
CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed --num_gpus=4 ./lerobot/scripts/dps_train_2.py \
    --deepspeed="./ds_zero2_40G.json" \
    --policy.type="pi05" \
    --policy.use_lora=true \
    --policy.loss_type="mse_loss" \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=true \
    --policy.add_new_tokens=false \
    --dataset.root="./data" \
    --dataset.repo_id="any/simulted" \
    --dataset.data_mix="cup_hz_4_plus_1103" \
    --dataset.image_transforms.enable=false \
    --wandb.enable=true \
    --wandb.project="CUPCUPCUP" \
    --job_name="0205-pi05-cup" \
    --log_dir="logs" \
    --output_dir="./amlt_output" \
    --steps=30_000 \
    --save_freq=5000 \
    --log_freq=20 \
    --policy.pt_weight_path="./model_new.pt" \
    # --policy.pt_weight_path="/Data/lzl/weights/pi_zero_pt/model.pt" \
    # --policy.pretrained_path=""/Data/lzl/pi0-ft-real/1124-latent-pi05-ft-real/step30000/mp_rank_00_model_states.pt"
    # --dataset.image_transforms.enable=true
    # "/Data/lzl/openpi/pytorch/pi05_base/model_new.pt"