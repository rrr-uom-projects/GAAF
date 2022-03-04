#!/bin/bash
python3 GAAF/preprocess.py --in_image_dir "/data/heartHunter/data/CTs"\ 
                              --in_mask_dir "/data/heartHunter/data/Structures"\
                              --out_image_dir "/data/heartHunter/data/rescaled_CTs"\
                              --CoM_targets_dir "/data/heartHunter/data"\
                              --Locator_image_resolution 64 128 128