from train import *

config = {
    "fusion_strategy": "early",
    "fusion_depth": "conv4_block6_out",
    "next_start_layer": "conv5_block1_1_conv",
    "fusion_method": "conv",
}

main(config)