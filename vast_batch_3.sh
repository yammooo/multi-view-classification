#!/bin/bash
# Run first training job
CUDA_VISIBLE_DEVICES=2 python multi-view-classification/vast_train_11_score.py

# Run second training job
CUDA_VISIBLE_DEVICES=2 python multi-view-classification/vast_train_13_score.py

# Wait 15 minutes (15*60 = 900 seconds)
sleep 1800

# Call the Vast API to stop the instance
vastai stop instance 19024669