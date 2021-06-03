python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone efficientnet_b0 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir efficientnet_b0_ft30 --weightdir efficientnet_b0 --fold 0 --seed 3024

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone efficientnet_b0 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir efficientnet_b0_ft30 --weightdir efficientnet_b0 --fold 1 --seed 2024

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnest50d_1s4x24d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d_ft30 --weightdir resnest50d_1s4x24d --fold 2 --seed 2023

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnest50d_1s4x24d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d_ft30 --weightdir resnest50d_1s4x24d --fold 3 --seed 3024

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnest50d_1s4x24d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d_ft30 --weightdir resnest50d_1s4x24d --fold 1 --seed 1023

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone ecaresnet26t --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir ecaresnet26t_ft30 --weightdir ecaresnet26t --fold 0 --seed 2021

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone ecaresnet26t --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir ecaresnet26t_ft30 --weightdir ecaresnet26t --fold 3 --seed 3023

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnest26d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnest26d_ft30 --weightdir resnest26d --fold 0 --seed 3023

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnest26d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnest26d_ft30 --weightdir resnest26d --fold 1 --seed 2024

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnest26d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnest26d_ft30 --weightdir resnest26d --fold 2 --seed 3025
 
python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone repvgg_b2 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir repvgg_b2_ft30 --weightdir repvgg_b2 --fold 2 --seed 3023

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone repvgg_b2 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir repvgg_b2_ft30 --weightdir repvgg_b2 --fold 3 --seed 2024

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnet50d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnet50d_ft30 --weightdir resnet50d --fold 1 --seed 1023

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone resnet50d --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir resnet50d_ft30 --weightdir resnet50d --fold 2 --seed 1024

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone efficientnet_b2 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir efficientnet_b2_ft30 --weightdir efficientnet_b2 --fold 3 --seed 1025

python main.py --gpus 3 --batch_size 36 --epochs 10 --backbone efficientnet_b2 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-4  --secondary_coef 1.0 --n_mels 128 --mixup_p 0.2 --mixup_alpha 0.8 --logdir efficientnet_b2_ft30 --weightdir efficientnet_b2 --fold 0 --seed 1026
