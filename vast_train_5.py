from train import *

config = {
    "fusion_strategy": "early",
    "fusion_depth": "conv2_block3_out",
    "next_start_layer": "conv3_block1_1_conv",
    "fusion_method": "conv",
}

main(config)