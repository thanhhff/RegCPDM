# train

# run with CPU
# python3 main.py --seed 1234 --config configs/RegCPDM.yaml --sample_to_eval --gpu_ids -1
# python3 main.py --seed 1234 --config configs/RegCPDM.yaml --train --sample_at_start --save_top --gpu_ids -1  --resume_model path/to/model.pth --resume_optim path/to/optimizer.pth

# run with GPU
# python3 main.py --seed 1234 --config configs/RegCPDM.yaml --train --sample_at_start --save_top --gpu_ids 0 --resume_model path/to/model.pth --resume_optim path/to/optimizer.pth
python3 main.py --seed 1234 --config configs/RegCPDM.yaml --train --sample_at_start --save_top --gpu_ids 2

# test

# run with CPU
# python3 main.py --config configs/RegCPDM.yaml --sample_to_eval --gpu_ids -1 --resume_model /path/to/resume/model/ckpt.pth
# run with GPU
# python3 main.py --config configs/RegCPDM.yaml --sample_to_eval --gpu_ids 1 --resume_model /path/to/resume/model/ckpt.pth