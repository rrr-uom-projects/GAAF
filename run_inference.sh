#!/bin/bash
python3 example_inference.py --model_dir "/home/ed/AutoGAF/models/fold2" \
                             --use_attention "False" \
                             --image_dir "/home/ed/AutoGAF/data/CTs" \
                             --output_dir "/home/ed/AutoGAF/data/test_cropped_cts" \
                             --subvolumes_or_coords "subvolumes" \
                             --Locator_resolution 64 128 128 \
                             --output_resolution 64 128 128