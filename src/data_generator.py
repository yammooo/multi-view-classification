import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image  # For cropping

class SimpleMultiViewDataGenerator:
    def __init__(self,
                 data_dir,
                 views=["back_left", "back_right", "front_left", "front_right", "top"],
                 input_shape=(512, 512, 3),
                 batch_size=8,
                 test_split=0.2,
                 random_state=42,
                 preprocess_fn=None,
                 crop_to_square=False):   # New parameter
        """
        Args:
            data_dir (str): Root directory with folders for each view.
            views (list): Names of view folders (e.g. ["view_1", "view_2", ...]).
            input_shape (tuple): Target image size.
            batch_size (int): Batch size.
            test_split (float): Fraction of data to reserve for testing.
            random_state (int): For reproducibility.
            preprocess_fn (function): A function that takes a numpy image array and returns a preprocessed array.
                                      For example, use tf.keras.applications.resnet50.preprocess_input.
                                      If None, it defaults to ResNet50’s preprocessing.
            crop_to_square (bool): If True, center-crop image to a square before resizing.
        """
        self.data_dir = data_dir
        self.views = views
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.test_split = test_split
        self.random_state = random_state
        self.preprocess_fn = preprocess_fn
        self.crop_to_square = crop_to_square  # Save the flag
        
        self._parse_dataset()
        self._split_dataset()

    def _parse_dataset(self):
        base_view = self.views[0]
        base_view_path = os.path.join(self.data_dir, base_view)
        if not os.path.isdir(base_view_path):
            raise ValueError(f"Base view folder not found: {base_view_path}")
        categories = [d for d in os.listdir(base_view_path) if os.path.isdir(os.path.join(base_view_path, d))]
        
        self.samples_by_category = {}
        for category in categories:
            sample_list = []
            base_cat_path = os.path.join(self.data_dir, base_view, category)
            base_files = [f for f in os.listdir(base_cat_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for filename in base_files:
                sample_views = []
                valid = True
                for view in self.views:
                    img_path = os.path.join(self.data_dir, view, category, filename)
                    if not os.path.exists(img_path):
                        valid = False
                        break
                    sample_views.append(img_path)
                if valid:
                    sample_list.append({
                        'category': category,
                        'views': sample_views
                    })
            if sample_list:
                self.samples_by_category[category] = sample_list
                
        self.category_list = sorted(self.samples_by_category.keys())
        self.category_to_idx = {cat: i for i, cat in enumerate(self.category_list)}
        total_samples = sum(len(lst) for lst in self.samples_by_category.values())
        print(f"Found {total_samples} complete samples for categories: {self.category_list}")

    def _split_dataset(self):
        """Split samples within every category so that every category appears in both train and test."""
        self.train_samples = []
        self.test_samples = []
        random.seed(self.random_state)
        for cat, samples in self.samples_by_category.items():
            random.shuffle(samples)
            n_test = int(round(len(samples) * self.test_split))
            # Ensure at least one sample in test if category has more than one sample.
            if n_test == 0 and len(samples) > 1:
                n_test = 1
            self.test_samples += samples[:n_test]
            self.train_samples += samples[n_test:]
    
    def _load_and_preprocess_image(self, img_path):
        """Load, optionally crop, resize, and preprocess an image."""
        # Load image without resizing first.
        img = load_img(img_path)
        
        # Optionally crop to square using center crop.
        if self.crop_to_square:
            width, height = img.size
            min_dim = min(width, height)
            left = (width - min_dim) // 2
            top = (height - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            img = img.crop((left, top, right, bottom))
            
        # Resize to target input shape.
        img = img.resize((self.input_shape[0], self.input_shape[1]))
        img_array = img_to_array(img)
        
        if self.preprocess_fn is None:
            return img_array
        return self.preprocess_fn(img_array)
    
    def _data_generator(self, samples):
        num_samples = len(samples)
        indices = list(range(num_samples))
        num_classes = len(self.category_list)
    
        while True:
            random.shuffle(indices)
            for start_idx in range(0, num_samples, self.batch_size):
                batch_indices = indices[start_idx:start_idx + self.batch_size]
                batch_samples = [samples[i] for i in batch_indices]
                batch_views = [np.zeros((len(batch_samples), *self.input_shape))
                               for _ in range(len(self.views))]
                batch_labels = np.zeros((len(batch_samples), num_classes))
                for i, sample in enumerate(batch_samples):
                    for v in range(len(self.views)):
                        img_path = sample['views'][v]
                        batch_views[v][i] = self._load_and_preprocess_image(img_path)
                    cat = sample['category']
                    batch_labels[i, self.category_to_idx[cat]] = 1.0
                yield tuple(batch_views), batch_labels

    def get_train_dataset(self):
        output_signature = (
            tuple([tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32)
                   for _ in range(len(self.views))]),
            tf.TensorSpec(shape=(None, len(self.category_to_idx)), dtype=tf.float32)
        )
        return tf.data.Dataset.from_generator(
            lambda: self._data_generator(self.train_samples),
            output_signature=output_signature
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def get_test_dataset(self):
        output_signature = (
            tuple([tf.TensorSpec(shape=(None, *self.input_shape), dtype=tf.float32)
                   for _ in range(len(self.views))]),
            tf.TensorSpec(shape=(None, len(self.category_to_idx)), dtype=tf.float32)
        )
        return tf.data.Dataset.from_generator(
            lambda: self._data_generator(self.test_samples),
            output_signature=output_signature
        ).prefetch(tf.data.experimental.AUTOTUNE)

    def get_class_names(self):
        return self.category_list

    def get_num_classes(self):
        return len(self.category_list)

    def visualize_batch(self):
        ds = self.get_train_dataset()
        views, labels = next(iter(ds))
        class_names = self.get_class_names()
        fig = plt.figure(figsize=(20, 10))
        for sample_idx in range(min(4, len(labels))):
            true_idx = np.argmax(labels[sample_idx])
            true_name = class_names[true_idx]
            for v in range(len(self.views)):
                image = views[v][sample_idx].numpy()
                image = np.clip(image, 0, 255).astype('uint8')
                ax = fig.add_subplot(4, len(self.views), sample_idx * len(self.views) + v + 1)
                ax.imshow(image)
                ax.text(5, 20, true_name, color='white', fontsize=12, backgroundcolor='black')
                if v == 0:
                    ax.set_ylabel(f"Sample {sample_idx+1}")
                if sample_idx == 0:
                    ax.set_title(f"View {self.views[v]}")
                ax.axis('off')
        plt.tight_layout()
        return fig

if __name__ == "__main__":
    # Example usage
    data_dir = r"/home/yammo/C:/Users/gianm/Development/blender-dataset-gen/data/output"
    views = ["back_left", "back_right", "front_left", "front_right", "top"]
    print("Creating data generator...")
    data_gen = SimpleMultiViewDataGenerator(
        data_dir=data_dir,
        views=views,
        input_shape=(512, 512, 3),
        batch_size=4,
        crop_to_square=True
    )
    
    fig = data_gen.visualize_batch()
    plt.show()