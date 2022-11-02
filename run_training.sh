#!/bin/bash
folds='1 2 3 4 5'
for fold_num in $folds
do
    python3 -m GAAF.train   --fold_num $fold_num \
                            --model_dir "/home/ed/GAAF/models/dummy" \
                            --image_dir "/home/ed/segmentation_work/GAAF_headHunter_train_data_HNSCC/rescaled_CTs" \
                            --CoM_targets_dir "/home/ed/segmentation_work/GAAF_headHunter_train_data_HNSCC" \
                            --train_BS 8 \
                            --val_BS 8 \
                            --train_workers 4 \
                            --val_workers 4
done