#!/bin/bash
folds='1 2 3 4 5'
for fold_num in $folds
do
    python3 -m GAAF.train   --fold_num $fold_num \
                            --model_dir "/path/to/directory/where/model/will/be/saved" \
                            --image_dir "/path/to/directory/where/processed/CTs/have/been/saved" \
                            --CoM_targets_dir "/path/to/directory/where/CoM_targets/file/has/been/saved" \
                            --train_BS 8 \
                            --val_BS 8 \
                            --train_workers 4 \
                            --val_workers 4
done