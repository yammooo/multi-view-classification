import wandb
wandb.login()

datasets = []

datasets.append("synt_real_75_5_dataset:v0")
datasets.append("test_edges:latest")
datasets.append("test_good_conditions:latest")
datasets.append("test_other_objects_interference:latest")
datasets.append("test_partial_occlusion:latest")

def download_dataset(label):
    # Initialize W&B
    wandb.init(project="5-view-classification", job_type="dataset-download")

    # Download dataset artifact
    artifact = wandb.use_artifact(f'yammo-unipd/5-view-classification/{label}', type='dataset')

    # Get the dataset directory
    dataset_dir = artifact.download()
    print(f"Dataset downloaded to: {dataset_dir}")

for dataset in datasets:
    download_dataset(dataset)

wandb.finish()
