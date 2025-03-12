import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import datetime
from data_generator import SimpleMultiViewDataGenerator
import sys

# Add the project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model_evaluation_utils import plot_training_history, evaluate_model, visualize_wrong_predictions
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
    data_dir = r"/home/yammo/C:/Users/gianm/Development/blender-dataset-gen/data/output"
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
    
    # Plot training history
    print("Plotting training history...")
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close(history_fig)
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)

    # Log the classification report as text and confusion matrix image to wandb
    wandb.log({
        "classification_report": report,
        "confusion_matrix": wandb.Image(cm_fig)
    })

    cm_fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(cm_fig)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    print("Visualizing model predictions...")
    pred_wrong_fig = visualize_wrong_predictions(model, test_ds, class_names)
    if pred_wrong_fig:
        pred_wrong_fig.savefig(os.path.join(output_dir, 'wrong_predictions.png'))
        plt.close(pred_wrong_fig)
    
    print(f"All results saved to {output_dir}")
    
    # Finish wandb run
    wandb.finish()
    
if __name__ == "__main__":
    main()