#!/bin/bash
python3 -m GAAF.preprocess  --in_image_dir "/path/to/directory/containing/nifti/CTs" \
                            --in_mask_dir "/path/to/directory/containing/nifti/masks/of/the/Structures" \
                            --out_image_dir "/path/to/directory/where/processed/CTs/will/be/saved" \
                            --CoM_targets_dir "/path/to/directory/where/CoM_targets/file/will/be/saved" \
                            --Locator_image_resolution 64 128 128 \
                            --target_ind 2