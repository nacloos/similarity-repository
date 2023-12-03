import os

for exp_name in ["feather", "layer_exp", "pca_deletion", "pretrain_finetune"]:
    os.system(f"cd {exp_name} && python script.py && cd ..")