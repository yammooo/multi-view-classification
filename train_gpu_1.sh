#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python multi-view-classification/train.py '{"backbone_model": "efficientnetb0", "fusion_strategy": "score", "fusion_depth": None, "next_start_layer": None, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["stem", "block1", "block2", "block3"]}}'