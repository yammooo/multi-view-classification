from train import *

config = {
    "fusion_strategy": "score",
    "fusion_depth": None,
    "next_start_layer": None,
    "fusion_method": "prod",
}

main(config)