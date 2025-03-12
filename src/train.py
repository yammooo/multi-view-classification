import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
from data_generator import SimpleMultiViewDataGenerator
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_evaluation_utils import *
from models.multi_view_model import build_multi_view_model
from models.early_fusion.resnet50_early import build_5_view_resnet50_early

# Import wandb and its Keras callbacks
import wandb
from wandb.integration.keras import WandbMetricsLogger, WandbModelCheckpoint

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # Initialize wandb
    wandb.init(
        project="5-view-classification",
        config={
            "dataset_artifact": "synt_5_obj_dataset:v0",
            "input_shape": (224, 224, 3),
            "batch_size": 8,
            "epochs": 1,
            "optimizer": "adam",
            "learning_rate": 1e-4,
            "backbone_model": "resnet50",
            "loss": "categorical_crossentropy",
            "fusion_type": "early",
            "fusion_depth": "conv2_block3_out",
            "fusion_method": "max",
        }
    )
    config = wandb.config
    
    # Data parameters
    data_dir = r"/home/yammo/C:/Users/gianm/Development/blender-dataset-gen/data/synt_5_obj_dataset_v0"
    input_shape = config.input_shape
    batch_size = config.batch_size
    
    # Initialize data generator
    print("Initializing data generator...")
    data_gen = SimpleMultiViewDataGenerator(
        data_dir=data_dir,
        input_shape=input_shape,
        batch_size=batch_size
    )
    
    # Get datasets
    train_ds = data_gen.get_train_dataset()
    test_ds = data_gen.get_test_dataset()
    class_names = data_gen.get_class_names()
    num_classes = data_gen.get_num_classes()
    
    # Create output directory for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize some augmented data
    print("Visualizing training data...")
    aug_fig = data_gen.visualize_batch()
    aug_fig.savefig(os.path.join(output_dir, "augmented_samples.png"))
    plt.close(aug_fig)
    
    # Build model
    print("Building multi-view model...")
    model = build_5_view_resnet50_early(
        input_shape=input_shape,
        num_classes=num_classes,
        fusion_type=config.fusion_method,
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(config.learning_rate),
        loss=config.loss,
        metrics=['accuracy']
    )
    
    # Create callbacks including wandb callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(output_dir, 'model_best.keras'),
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        WandbMetricsLogger(log_freq=5),
        WandbModelCheckpoint(os.path.join(output_dir, "wandb_model_best.keras"))
    ]
        
    # Compute steps per epoch based on the flattened sample lists (train_samples, test_samples)
    steps_per_epoch = len(data_gen.train_samples) // batch_size
    if len(data_gen.train_samples) % batch_size != 0:
        steps_per_epoch += 1
    
    validation_steps = len(data_gen.test_samples) // batch_size
    if len(data_gen.test_samples) % batch_size != 0:
        validation_steps += 1

    # Update wandb config with computed steps
    wandb.config.update({
        "steps_per_epoch": steps_per_epoch,
        "validation_steps": validation_steps
    })

    # Then pass these to model.fit so the epoch stops once the dataset is exhausted:
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Save the final model using the native Keras format
    model.save(os.path.join(output_dir, 'model_final.keras'))
        
    # Evaluate model
    print("Evaluating model on test dataset...")
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)
    pred_wrong_fig = visualize_wrong_predictions(model, test_ds, class_names)

    # Log the classification report as text and confusion matrix image to wandb
    wandb.log({
        "classification_report": format_classification_report(report, class_names),
        "pred_wrong": wandb.Image(pred_wrong_fig) if pred_wrong_fig is not None else None,
        "confusion_matrix": wandb.Image(cm_fig)
    })
        
    # Finish wandb run
    wandb.finish()
    
if __name__ == "__main__":
    main()