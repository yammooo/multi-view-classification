#!/bin/bash

CUDA_VISIBLE_DEVICES=1 python multi-view-classification/train.py '{"backbone_model": "resnet152", "fusion_strategy": "score", "fusion_depth": None, "next_start_layer": None, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["conv1", "conv2", "conv3"]}}'