#!/bin/bash
python3 -m GAAF.preprocess  --in_image_dir "/home/ed/segmentation_work/big_HNSCC_dataset/data/CTs"\
                            --in_mask_dir "/home/ed/segmentation_work/big_HNSCC_dataset/data/Structures"\
                            --out_image_dir "/home/ed/segmentation_work/GAAF_headHunter_train_data_HNSCC/dummy"\
                            --CoM_targets_dir "/home/ed/segmentation_work/dummy"\
                            --Locator_image_resolution 64 128 128\
                            --target_ind 2