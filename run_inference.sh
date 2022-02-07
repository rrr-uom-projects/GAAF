#!/bin/bash
python3 example_inference.py --model_dir "\data\ed\AutoGAF\models\fold2" \
                             --use_attention "False" \
                             --image_dir "" \
                             --output_dir "" \
                             --subvolumes_or_coords "subvolumes" \
                             --Locator_resolution 64 128 128 \
                             --output_resolution 64 128 128