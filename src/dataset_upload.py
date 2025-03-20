import json
import wandb
import os

DATASET_DIR = r"/home/yammo/Downloads/synt+real_75+5_dataset_v0.rar"

# synt_5_obj_dataset, real_obj_dataset
DATASET_NAME = "synt_real_75_5_dataset"

# Read render parameters from JSON file
if os.path.exists(os.path.join(DATASET_DIR, "config.json")):
    with open(os.path.join(DATASET_DIR, "config.json"), "r") as f:
        render_params = json.load(f)
else:
    render_params = {}

DESCRIPTION = "A synthetic dataset generated from Blender with 5 camera views. It contains 80 classes: 5 from committee, 75 from thingi10k. For each object, 750 256x256 5-view images were generated."
METADATA = {
                "source": "Blender Dataset Generator",
                "render_params": render_params
            }

#DESCRIPTION = None
#METADATA = None

# Initialize project
wandb.init(project="5-view-classification", job_type="dataset-upload")

# Create dataset artifact with render parameters as part of metadata
artifact = wandb.Artifact(DATASET_NAME, type="dataset",
                          description=DESCRIPTION,
                          metadata=METADATA)

artifact.add_file(DATASET_DIR)
#artifact.add_dir(DATASET_DIR)

# Log dataset
wandb.log_artifact(artifact)
wandb.finish()