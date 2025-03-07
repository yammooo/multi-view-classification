import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re


class MultiViewDataGenerator:
    def __init__(self, 
                 data_dir, 
                 input_shape=(224, 224, 3),
                 batch_size=8,
                 test_split=0.2,
                 random_state=42,
                 num_views=5):
        """
        Initialize the Multi-View Data Generator.
        
        Args:
            data_dir (str): Path to the dataset directory
            input_shape (tuple): Input shape for the model (height, width, channels)
            batch_size (int): Batch size
            test_split (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            num_views (int): Number of views per object (default 5)
        """
        self.data_dir = data_dir
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.test_split = test_split
        self.random_state = random_state
        self.num_views = num_views
        
        # Create augmentation layers
        self.augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomFlip("horizontal"),
            #tf.keras.layers.RandomRotation(0.1),
            #tf.keras.layers.RandomZoom(0.1),
            #tf.keras.layers.RandomContrast(0.2),
            #tf.keras.layers.RandomBrightness(0.2),
            #tf.keras.layers.RandomTranslation(0.1, 0.1),
            #tf.keras.layers.RandomHue(0.1),
            #tf.keras.layers.RandomSaturation(0.2),
        ])

        # Parse dataset structure and prepare data
        self._parse_dataset()
        self._split_dataset()
        
    def _extract_uid(self, filename):
        """Extract unique object ID from filename."""
        match = re.match(r'([^_]+)_v\d+\.png', filename)
        if match:
            return match.group(1)
        return None
    
    def _parse_dataset(self):
        """Parse dataset structure and group by UIDs."""
        self.categories = {}
        self.category_to_idx = {}
        self.objects_by_uid = {}
        
        # Get all categories
        categories = sorted([d for d in os.listdir(self.data_dir) 
                             if os.path.isdir(os.path.join(self.data_dir, d))])
        
        for idx, category in enumerate(categories):
            self.category_to_idx[category] = idx
            cat_dir = os.path.join(self.data_dir, category)
            
            # Check if all view directories exist
            view_dirs = ["back_left", "back_right", "front_left", "front_right", "top"]
            
            # Process each view directory
            for view_idx, view_dir in enumerate(view_dirs):
                view_path = os.path.join(cat_dir, view_dir)
                for img_file in os.listdir(view_path):
                    if not img_file.endswith('.png'):
                        continue
                    
                    uid = self._extract_uid(img_file)
                    if not uid:
                        continue
                    
                    # Create entry for this UID if it doesn't exist
                    if uid not in self.objects_by_uid:
                        self.objects_by_uid[uid] = {
                            'category': category,
                            'category_idx': idx,
                            'views': [[] for _ in range(self.num_views)]
                        }
                    
                    # Add image path for this view
                    img_path = os.path.join(view_path, img_file)
                    self.objects_by_uid[uid]['views'][view_idx].append(img_path)
        
        # Filter out objects that don't have all views
        self.complete_uids = [uid for uid, data in self.objects_by_uid.items() 
                             if all(data['views'])]
        
        print(f"Found {len(categories)} categories")
        print(f"Found {len(self.complete_uids)} complete objects with all {self.num_views} views")
        print(f"Filtered out {len(self.objects_by_uid) - len(self.complete_uids)} objects with missing views")
        
    def _split_dataset(self):
        """Split dataset into training and testing by UIDs."""
        # Split UIDs into training and testing sets
        train_uids, test_uids = train_test_split(
            self.complete_uids,
            test_size=self.test_split,
            random_state=self.random_state,
            stratify=[self.objects_by_uid[uid]['category'] for uid in self.complete_uids]
        )
        
        self.train_uids = train_uids
        self.test_uids = test_uids
        
        print(f"Training set: {len(train_uids)} objects")
        print(f"Testing set: {len(test_uids)} objects")
        
        # Expand each UID into individual training samples according to the number of versions.
        self.train_samples = []
        self.test_samples = []
        for uid in self.train_uids:
            obj_data = self.objects_by_uid[uid]
            # Use the minimum number of versions among all views (to ensure every view has a corresponding image)
            n_versions = min(len(v_list) for v_list in obj_data['views'])
            for version in range(n_versions):
                self.train_samples.append((uid, version))
    
        for uid in self.test_uids:
            obj_data = self.objects_by_uid[uid]
            n_versions = min(len(v_list) for v_list in obj_data['views'])
            for version in range(n_versions):
                self.test_samples.append((uid, version))
   
    def _load_and_preprocess_image(self, img_path, augment=False):
        """Load and preprocess an image."""
        img = load_img(img_path, target_size=(self.input_shape[0], self.input_shape[1]))
        img_array = img_to_array(img)
        
        if augment:
            # Add batch dimension for augmentation
            img_array = tf.expand_dims(img_array, 0)
            img_array = self.augmentation(img_array)
            img_array = tf.squeeze(img_array, 0)
        
        # Apply ResNet50 preprocessing (subtract mean RGB values)
        img_array = preprocess_input(img_array)
        return img_array
    
    def _data_generator(self, samples, augment=False):
        """Create a generator that yields batches of multi-view data.
        samples is a list of (uid, version) tuples.
        """
        num_samples = len(samples)
        indices = list(range(num_samples))
        num_classes = len(self.category_to_idx)
        
        while True:
            random.shuffle(indices)
            
            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_samples = [samples[i] for i in batch_indices]
                
                # Initialize arrays for views and labels. (One sample = 5 views)
                batch_views = [np.zeros((len(batch_samples), *self.input_shape)) 
                            for _ in range(self.num_views)]
                batch_labels = np.zeros((len(batch_samples), num_classes))
                
                for i, (uid, version) in enumerate(batch_samples):
                    obj_data = self.objects_by_uid[uid]
                    
                    # For each view, load the image corresponding to the given version
                    for view_idx in range(self.num_views):
                        img_path = obj_data['views'][view_idx][version]
                        batch_views[view_idx][i] = self._load_and_preprocess_image(img_path, augment=augment)
                    
                    # One-hot encode the label
                    batch_labels[i, obj_data['category_idx']] = 1.0
                
                yield tuple(batch_views), batch_labels
    
    def get_train_dataset(self, augment=True):
        output_signature = (
            tuple([tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32) 
                for _ in range(self.num_views)]),
            tf.TensorSpec(shape=(None, len(self.category_to_idx)), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(self.train_samples, augment=augment),
            output_signature=output_signature
        )
        
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    def get_test_dataset(self):
        output_signature = (
            tuple([tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32) 
                for _ in range(self.num_views)]),
            tf.TensorSpec(shape=(None, len(self.category_to_idx)), dtype=tf.float32)
        )
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self._data_generator(self.test_samples, augment=False),
            output_signature=output_signature
        )
        
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)    

    def get_class_names(self):
        """Return a list of class names in index order."""
        return [cat for cat, idx in sorted(self.category_to_idx.items(), key=lambda x: x[1])]
    
    def get_num_classes(self):
        """Return the number of classes in the dataset."""
        return len(self.category_to_idx)
    
    def visualize_batch(self, augmented=True):
        """Visualize a batch of data to check augmentation."""
        dataset = self.get_train_dataset(augment=augmented)
        views, labels = next(iter(dataset))
        
        class_names = self.get_class_names()
        
        # Create a figure to display images
        fig = plt.figure(figsize=(20, 10))
        
        # Display 4 samples with all 5 views each
        for sample_idx in range(min(4, len(labels))):
            class_idx = np.argmax(labels[sample_idx])
            class_name = class_names[class_idx]
            
            for view_idx in range(self.num_views):
                # Get the image and convert from preprocessed back to display format
                image = views[view_idx][sample_idx].numpy()
                
                # Reverse preprocess_input for display
                # For ResNet50: Add back the mean values subtracted during preprocessing
                image = image + np.array([103.939, 116.779, 123.68])  # BGR means for ResNet50
                image = image[..., ::-1]  # BGR to RGB
                image = np.clip(image, 0, 255).astype('uint8')
                
                # Create subplot
                ax = fig.add_subplot(4, self.num_views, sample_idx * self.num_views + view_idx + 1)
                ax.imshow(image)
                
                if view_idx == 0:
                    ax.set_ylabel(f"Sample {sample_idx+1}\n{class_name}")
                
                if sample_idx == 0:
                    ax.set_title(f"View {view_idx+1}")
                
                ax.axis('off')
        
        plt.tight_layout()
        plt.suptitle("Multi-View Images " + ("(with augmentation)" if augmented else "(original)"), y=1.02)
        return fig


if __name__ == "__main__":
    # Example usage
    data_dir = r"/home/yammo/C:/Users/gianm/Development/multi-view-classification/dataset"
    print("Creating data generator...")
    data_gen = MultiViewDataGenerator(
        data_dir=data_dir,
        input_shape=(512, 512, 3),
        batch_size=4
    )
    
    # Visualize augmented images
    fig = data_gen.visualize_batch(augmented=True)
    plt.show()  # This displays the plot window