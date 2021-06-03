python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnet34 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnet34 --fold 0 --seed 2021

python main.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnet34 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnet34 --fold 1 --seed 2022

python main_v4.py --gpus 3 --batch_size 36 --epochs 40 --backbone resnet34 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir resnet34 --fold 2 --seed 3023

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone efficientnet_b0 --period 20.0 --infer_period 20.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir efficientnet_b0 --fold 0 --seed 2024

python main.py --gpus 3 --batch_size 36 --epochs 50 --backbone efficientnet_b0 --period 20.0 --infer_period 20.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir efficientnet_b0 --fold 2 --seed 2024

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnest50d_1s4x24d --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d --fold 2 --seed 2023

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnest50d_1s4x24d --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d --fold 3 --seed 2024

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnest50d_1s4x24d --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d --fold 1 --seed 4023

python main_v4.py --gpus 3 --batch_size 36 --epochs 40 --backbone resnest50d_1s4x24d --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir resnest50d_1s4x24d --fold 0 --seed 3025

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnet18 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnet18 --fold 3 --seed 2024

python main_v4.py --gpus 3 --batch_size 36 --epochs 40 --backbone resnet18 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir resnet18 --fold 0 --seed 3026

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone ecaresnet26t --period 15.0 --infer_period 15.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir ecaresnet26t --fold 3 --seed 2023

python main_v4.py --gpus 3 --batch_size 36 --epochs 40 --backbone ecaresnet26t --period 15.0 --infer_period 15.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir ecaresnet26t --fold 0 --seed 3027

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone mixnet_m --period 15.0 --infer_period 15.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir mixnet_m --fold 1 --seed 2023

python main_v4.py --gpus 3 --batch_size 36 --epochs 50 --backbone mixnet_m --period 15.0 --infer_period 15.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir mixnet_m --fold 2 --seed 2024

python main_v2.py --gpus 3 --batch_size 36 --epochs 40 --backbone mixnet_m --period 15.0 --infer_period 15.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir mixnet_m --fold 3 --seed 3028

python main_v4.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnest26d --period 20.0 --infer_period 20.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnest26d --fold 0 --seed 2023

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnest26d --period 20.0 --infer_period 20.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnest26d --fold 1 --seed 2024

python main_v4.py --gpus 3 --batch_size 36 --epochs 40 --backbone resnest26d --period 20.0 --infer_period 20.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir resnest26d --fold 2 --seed 3029

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone repvgg_b0 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir repvgg_b0 --fold 2 --seed 2023

python main_v4.py --gpus 3 --batch_size 36 --epochs 50 --backbone repvgg_b0 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir repvgg_b0 --fold 3 --seed 2024

python main_v2.py --gpus 3 --batch_size 36 --epochs 40 --backbone repvgg_b0 --period 30.0 --infer_period 30.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir repvgg_b0 --fold 0 --seed 3030

python main_v4.py --gpus 3 --batch_size 36 --epochs 50 --backbone repvgg_b2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir repvgg_b2 --fold 2 --seed 2023

python main_v2.py --gpus 3 --batch_size 36 --epochs 50 --backbone repvgg_b2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir repvgg_b2 --fold 3 --seed 2024

python main_v4.py --gpus 3 --batch_size 36 --epochs 40 --backbone repvgg_b2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.1 --mixup_alpha 0.8 --logdir repvgg_b2 --fold 1 --seed 3031

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnet50d --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnet50d --fold 1 --seed 4023

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone resnet50d --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir resnet50d --fold 2 --seed 4024

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone efficientnet_b2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir efficientnet_b2 --fold 3 --seed 4025

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone efficientnet_b2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir efficientnet_b2 --fold 0 --seed 4026

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone wide_resnet50_2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir wide_resnet50_2 --fold 0 --seed 4025

python main_v5.py --gpus 3 --batch_size 36 --epochs 50 --backbone wide_resnet50_2 --period 10.0 --infer_period 10.0 --num_workers 4 --lr 1e-3 --backbone_lr 1e-4 --secondary_coef 1.0 --n_mels 128 --mixup_p 0.5 --mixup_alpha 0.8 --logdir wide_resnet50_2 --fold 2 --seed 4026