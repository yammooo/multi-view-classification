import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import datetime
from data_generator import MultiViewDataGenerator
import sys

# Add the project root directory to Python path
sys.path.append("/home/yammo/C:/Users/gianm/Development/multi-view-classification")
from model_evaluation import plot_training_history, evaluate_model, visualize_wrong_predictions
from models.multi_view_model import build_multi_view_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def main():
    # Data parameters
    data_dir = r"/home/yammo/C:/Users/gianm/Development/multi-view-classification/dataset"
    input_shape = (224, 224, 3)
    batch_size = 8
    
    # Initialize data generator
    print("Initializing data generator...")
    data_gen = MultiViewDataGenerator(
        data_dir=data_dir,
        input_shape=input_shape,
        batch_size=batch_size
    )
    
    # Get datasets
    train_ds = data_gen.get_train_dataset(augment=True)
    test_ds = data_gen.get_test_dataset()
    class_names = data_gen.get_class_names()
    num_classes = data_gen.get_num_classes()
    
    # Create output directory for results
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"results/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize some augmented data
    print("Visualizing training data...")
    aug_fig = data_gen.visualize_batch(augmented=True)
    aug_fig.savefig(os.path.join(output_dir, "augmented_samples.png"))
    plt.close(aug_fig)
    
    # Build model
    print("Building multi-view model...")
    model = build_multi_view_model(
        input_shape=input_shape,
        num_classes=num_classes,
        fusion_type='fc'  # or 'max'
    )
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Create callbacks
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
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(output_dir, 'logs')
        )
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
        epochs=1,
        callbacks=callbacks,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps
    )
    
    # Save the final model
    model.save(os.path.join(output_dir, 'model_final.keras'))
    
    # Plot training history
    print("Plotting training history...")
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close(history_fig)
    

    # Evaluate model
    print("Evaluating model on test dataset...")
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)
    cm_fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(cm_fig)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Visualize predictions
    print("Visualizing model predictions...")
    pred_wrong_fig = visualize_wrong_predictions(model, test_ds, class_names)
    if pred_wrong_fig:
        pred_wrong_fig.savefig(os.path.join(output_dir, 'wrong_predictions.png'))
        plt.close(pred_wrong_fig)
    
    print(f"All results saved to {output_dir}")
    
if __name__ == "__main__":
    main()