#!/bin/bash
# Run first training job
CUDA_VISIBLE_DEVICES=1 python multi-view-classification/vast_train_10_late.py

# Run second training job
CUDA_VISIBLE_DEVICES=1 python multi-view-classification/vast_train_3.py
