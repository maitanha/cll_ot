python scl-train.py \
    --algo=scl-fwd \
    --dataset_name=cifar10 \
    --model=resnet18 \
    --lr=1e-4 \
    --seed=0 \
    --data_aug=false \
    --batch_size 256 \
    --evaluate_step 5 \
    --n_epoch 300