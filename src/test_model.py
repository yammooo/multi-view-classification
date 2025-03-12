import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from matplotlib import pyplot as plt
import tensorflow as tf
from data_generator import SimpleMultiViewDataGenerator
from model_evaluation_utils import plot_training_history, evaluate_model, visualize_wrong_predictions
from models.early_fusion.resnet50_early import build_5_view_resnet50_early
import math
import keras
keras.config.enable_unsafe_deserialization()

# Load your already trained model (or build & load weights as needed)
model = tf.keras.models.load_model(r"/home/yammo/C:/Users/gianm/Development/multi-view-classification/results/run_20250312-084945/model_final.keras")

# Set your test dataset directory
test_data_dir = r"/home/yammo/C:/Users/gianm/Development/multi-view-classification/dataset/test"

test_gen = SimpleMultiViewDataGenerator(
    data_dir=test_data_dir,
    views=["back_left", "back_right", "front_left", "front_right", "top"],
    input_shape=(224, 224, 3),
    batch_size=8,
    test_split=1.0
)

# Get the test dataset (you can call get_test_dataset; or since the entire set is test,
# you might create a dataset from all samples)
test_ds = test_gen.get_test_dataset()
class_names = test_gen.get_class_names()
num_classes = test_gen.get_num_classes()

validation_steps = math.ceil(len(test_gen.test_samples) / test_gen.batch_size)

# Evaluate model
print("Evaluating model on test dataset...")
report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)

# Visualize wrong predictions
print("Visualizing wrong predictions...")
wrong_fig = visualize_wrong_predictions(model, test_ds, class_names, num_samples=5, visualize_original=True)
if wrong_fig is not None:
    # Optionally, save the figure or display it
    wrong_fig.savefig("wrong_predictions.png")
    plt.show()