#!/bin/bash
folds='1 2 3 4 5'
for fold_num in $folds
do
    python3 AutoGAF/train.py --fold_num $fold_num \
                             --model_dir "/home/ed/AutoGAF/models/attention" \
                             --image_dir "/home/ed/AutoGAF/data/rescaled_CTs" \
                             --CoM_targets_dir "/home/ed/AutoGAF/data" \
                             --train_BS 8 \
                             --val_BS 8 \
                             --train_workers 4 \
                             --val_workers 4
done