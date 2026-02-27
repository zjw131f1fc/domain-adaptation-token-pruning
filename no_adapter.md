outputs/tasks/20260227-0811_vqa-vqav2_llava157b_5df4/checkpoints/checkpoint_final.pt


python scripts/visualize_distribution_shift_v2.py \
    --checkpoint outputs/tasks/20260227-0811_vqa-vqav2_llava157b_5df4/checkpoints/checkpoint_final.pt \
    --config configs/vision_token_pruning.yaml \
    --num_samples 50 \
    --device cuda:0 \
    --mode gap_curve \
    --proj_dim 64 \
    --force_no_adapter