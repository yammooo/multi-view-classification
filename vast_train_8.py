from train import *

config = {
    "fusion_strategy": "early",
    "fusion_depth": "conv5_block3_out",
    "next_start_layer": None,
    "fusion_method": "conv",
}

main(config)