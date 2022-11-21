#!/bin/bash
python3 -m GAAF.inference   --model_dir "/path/to/directory/where/model/has/been/saved" \
                            --use_attention "True" \
                            --Locator_image_resolution 64 128 128 \
                            --in_image_dir "/path/to/directory/containing/nifti/CTs" \
                            --in_mask_dir "/path/to/directory/containing/nifti/masks/of/the/Structures" \
                            --output_crop True \
                            --out_image_dir "/path/to/directory/where/processed/CTs/will/be/saved" \
                            --out_mask_dir "/path/to/directory/where/processed/Structures/will/be/saved" \
                            --cropped_image_resolution 64 200 200 \
                            --output_coords True \
                            --out_CoMs_dir "/path/to/directory/where/CoM_predictions/file/will/be/saved" \
                            --cropped_image_slice_thickness 3.0