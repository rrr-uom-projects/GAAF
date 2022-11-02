#!/bin/bash
folds='1 2 3 4 5'
for fold_num in $folds
do
    python3 -m GAAF.test    --fold_num $fold_num \
                            --model_dir "/path/to/directory/where/model/has/been/saved" \
                            --image_dir "/path/to/directory/where/processed/CTs/have/been/saved" \
                            --CoM_targets_dir "/path/to/directory/where/CoM_targets/file/has/been/saved"
done