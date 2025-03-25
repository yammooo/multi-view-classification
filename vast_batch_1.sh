#!/bin/bash
# Run first training job
CUDA_VISIBLE_DEVICES=0 python multi-view-classification/vast_train_9_late.py

# Run second training job
CUDA_VISIBLE_DEVICES=0 python multi-view-classification/vast_train_2.py
