import os
import shutil
from argparse import ArgumentParser

def create_dataset_from_photos(input_dir, output_dir, category):
    """
    Creates a dataset with the following structure:
    
    output/
    │
    ├── back_left/
    │   └── <category>/
    │       ├── <category>_v1.png
    │       ├── <category>_v2.png
    │       └── ...
    ├── back_right/
    │   └── <category>/
    ├── front_left/
    │   └── <category>/
    ├── front_right/
    │   └── <category>/
    └── top/
        └── <category>/
    
    The input folder contains images sorted by name. They repeat in cycles of 5 with the 
    order: back_left, back_right, front_left, front_right, top.
    
    Args:
        input_dir (str): Path to the folder containing all photos.
        output_dir (str): Path where the organized dataset will be created.
        category (str): The category name for these photos.
    """
    # Define view names in the required order
    views = ["back_left", "back_right", "front_left", "front_right", "top"]
    
    # Create output directories for each view and category
    for view in views:
        category_dir = os.path.join(output_dir, view, category)
        os.makedirs(category_dir, exist_ok=True)
    
    # Get a sorted list of files from the input directory
    files = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])
    
    # Create a counter for each view
    counts = {view: 0 for view in views}
    
    # Loop through files, assign each to a view in a cyclic order
    for i, filename in enumerate(files):
        view = views[i % len(views)]
        counts[view] += 1
        new_filename = f"{category}_v{counts[view]}.png"
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, view, category, new_filename)
        shutil.copy(src_path, dst_path)
    
    print("Dataset created successfully.")

if __name__ == "__main__":

    base_folder_name = "charging_brick"
    input_dir = f"/home/yammo/Downloads/{base_folder_name}"
    output_dir = f"/home/yammo/Development/multi-view-classification/dataset/test_{base_folder_name}"

    # input_dir = r"/home/yammo/Downloads/other_objects_interference"
    # output_dir = r"/home/yammo/Development/multi-view-classification/dataset/test_other_objects_interference"

    category = "real_1"
    
    create_dataset_from_photos(input_dir, output_dir, category)