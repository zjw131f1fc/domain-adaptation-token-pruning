llava-13b:  outputs/tasks/20260224-1402_vqa-vqav2_llava1513b_5d39/checkpoints/checkpoint_final.pt

mse: outputs/tasks/20260224-1820_vqa-vqav2_llava157b_9931/checkpoints/checkpoint_final.pt

sqa 192: outputs/tasks/20260225-1708_vqa-sqa_llava157b_1152/checkpoints/checkpoint_final.pt

no adapter only pruner  128实际约192  outputs/tasks/20260227-0811_vqa-vqav2_llava157b_5df4/checkpoints/checkpoint_final.pt
 Avg kept ratio: 34.67%


192 vqav2base  outputs/tasks/20260301-1919_vqa-vqav2_llava157b_c110/checkpoints/checkpoint_final.pt

128 vqav2base  outputs/tasks/20260302-0000_vqa-vqav2_llava157b_acea/checkpoints/checkpoint_final.pt

64 vqav2base  outputs/tasks/20260301-2248_vqa-vqav2_llava157b_250c/checkpoints/checkpoint_final.pt

64 vqav2base no adapter outputs/tasks/20260302-0853_vqa-vqav2_llava157b_c65b/checkpoints/checkpoint_final.pt   76.73/80.49

64 vqav2base no adapter no repair loss outputs/tasks/20260302-0945_vqa-vqav2_llava157b_882d/checkpoints/checkpoint_final.pt  76.73  不要了  看起来repair loss 可以看作只作用在adapter上，也可能是梯度传到有bug，传不到pruner

64 vqav2base no repair loss outputs/tasks/20260302-1035_vqa-vqav2_llava157b_8507/checkpoints/checkpoint_final.pt  74.83



outputs/tasks/20260304-1214_vqa-vqav2_llava157b_dbae/checkpoints/checkpoint_final.pt


outputs/tasks/20260304-1405_vqa-vqav2_llava157b_6056/checkpoints/checkpoint_final.pt