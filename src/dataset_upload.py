import json
import wandb
import os

DATASET_DIR = r"/home/yammo/C:/Users/gianm/Development/blender-dataset-gen/data/output"

# Read render parameters from JSON file
with open(os.path.join(DATASET_DIR, "config.json"), "r") as f:
    render_params = json.load(f)

# print(render_params)

# Initialize project
wandb.init(project="5-view-classification", job_type="dataset-upload")

# Create dataset artifact with render parameters as part of metadata
artifact = wandb.Artifact("synt_5_obj_dataset", type="dataset",
                          description="A synthetic dataset generated from Blender with 5 camera views. It contains 5 classes of the objects of the committee. For each object, 1500 5-view images were generated.",
                          metadata={
                               "source": "Blender Dataset Generator",
                               "render_params": render_params
                           })

artifact.add_dir(DATASET_DIR)

# Log dataset
wandb.log_artifact(artifact, aliases=["latest", "v1.0"])
wandb.finish()