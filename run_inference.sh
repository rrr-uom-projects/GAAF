#!/bin/bash
python3 GAAF/inference.py --model_dir "/home/ed/GAAF/models/headHunter2/fold1/" \
                          --use_attention "True" \
                          --in_image_dir "/mnt/driger/hn_data/Aza_HnN/CTs" \
                          --out_image_dir "/mnt/driger/hn_data/Aza_HnN/headHunted/CTs" \
                          --out_CoMs_dir "None"\
                          --in_mask_dir "/mnt/driger/hn_data/Aza_HnN/Structures" \
                          --out_mask_dir "/mnt/driger/hn_data/Aza_HnN/headHunted/Structures" \
                          --Locator_image_resolution 64 128 128 \
                          --cropped_image_resolution 64 200 200 \
                          --cropped_image_slice_thickness 3.0