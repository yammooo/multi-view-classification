#!/bin/bash

CUDA_VISIBLE_DEVICES=3 python multi-view-classification/train.py '{"backbone_model": "convnextbase", "fusion_strategy": "score", "fusion_depth": null, "next_start_layer": null, "fusion_method": "sum", "freeze_config": {"freeze_blocks": ["stem", "stage_0"]}}'