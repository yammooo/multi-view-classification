import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import datetime
from data_generator import MultiViewDataGenerator
import sys
sys.path.append("..")
from models.multi_view_model import build_multi_view_model

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def plot_training_history(history):
    """Plot the training and validation accuracy/loss curves."""
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Epoch')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    return plt.gcf()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    return plt.gcf()

def evaluate_model(model, test_dataset, class_names):
    """Evaluate model on the test dataset."""
    # Collect all predictions and true labels
    y_true = []
    y_pred = []
    
    for views, labels in test_dataset:
        # Get predictions
        batch_preds = model.predict(views)
        
        # Convert to class indices
        batch_pred_classes = np.argmax(batch_preds, axis=1)
        batch_true_classes = np.argmax(labels.numpy(), axis=1)
        
        # Append to lists
        y_true.extend(batch_true_classes)
        y_pred.extend(batch_pred_classes)
    
    # Generate classification report
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    cm_fig = plot_confusion_matrix(y_true, y_pred, class_names)
    
    return report, cm_fig, y_true, y_pred

def visualize_predictions(model, test_dataset, class_names, num_samples=5):
    """Visualize model predictions on test samples."""
    for views, labels in test_dataset.take(1):  # Take one batch
        # Get predictions
        preds = model.predict(views)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        
        # Create figure
        fig = plt.figure(figsize=(20, 4 * min(num_samples, len(views[0]))))
        
        for i in range(min(num_samples, len(views[0]))):
            # Get true and predicted classes
            true_class = true_classes[i]
            pred_class = pred_classes[i]
            
            # Display all views of this sample
            for v in range(len(views)):
                # Reverse preprocess_input for display
                image = views[v][i].numpy()
                image = image + np.array([103.939, 116.779, 123.68])  # BGR means for ResNet50
                image = image[..., ::-1]  # BGR to RGB
                image = np.clip(image, 0, 255).astype('uint8')
                
                ax = fig.add_subplot(num_samples, len(views), i*len(views) + v + 1)
                ax.imshow(image)
                
                if v == 0:
                    ax.set_ylabel(f"Sample {i+1}")
                
                if i == 0:
                    ax.set_title(f"View {v+1}")
                
                ax.axis('off')
            
            # Add prediction result
            result_text = f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}"
            color = 'green' if true_class == pred_class else 'red'
            fig.text(0.02, 0.9 - (i * 0.2), result_text, fontsize=12, color=color)
        
        plt.tight_layout()
        return fig

def main():
    # Data parameters
    data_dir = "../data"  # Update to your dataset path
    input_shape = (512, 512, 3)
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
    output_dir = f"../results/run_{timestamp}"
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
            filepath=os.path.join(output_dir, 'model_best.h5'),
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
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=50,
        callbacks=callbacks
    )
    
    # Save the final model
    model.save(os.path.join(output_dir, 'model_final.h5'))
    
    # Plot training history
    print("Plotting training history...")
    history_fig = plot_training_history(history)
    history_fig.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close(history_fig)
    
    # Evaluate model
    print("Evaluating model on test dataset...")
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names)
    cm_fig.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close(cm_fig)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Visualize predictions
    print("Visualizing model predictions...")
    pred_fig = visualize_predictions(model, test_ds, class_names)
    pred_fig.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close(pred_fig)
    
    print(f"All results saved to {output_dir}")
    
    # Phase 2: Fine-tuning (optional)
    print("Starting fine-tuning phase...")
    
    # Unfreeze some layers for fine-tuning
    # For each branch model, unfreeze the last convolutional block
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # This is a branch model
            # Get the base model inside the branch
            base_model = layer.layers[0]
            # Unfreeze the last few layers (e.g., last 20 layers)
            for i in range(len(base_model.layers)-20, len(base_model.layers)):
                if hasattr(base_model.layers[i], 'trainable'):
                    base_model.layers[i].trainable = True
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Continue training
    fine_tune_history = model.fit(
        train_ds,
        validation_data=test_ds,
        epochs=30,
        callbacks=[
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(output_dir, 'model_fine_tuned.h5'),
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
            )
        ]
    )
    
    # Evaluate fine-tuned model
    print("Evaluating fine-tuned model...")
    report_ft, cm_fig_ft, _, _ = evaluate_model(model, test_ds, class_names)
    cm_fig_ft.savefig(os.path.join(output_dir, 'confusion_matrix_fine_tuned.png'))
    plt.close(cm_fig_ft)
    
    with open(os.path.join(output_dir, 'classification_report_fine_tuned.txt'), 'w') as f:
        f.write(report_ft)
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()