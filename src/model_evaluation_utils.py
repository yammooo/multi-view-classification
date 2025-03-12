import wandb
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import datetime
from data_generator import SimpleMultiViewDataGenerator
import sys


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

def evaluate_model(model, test_dataset, class_names, validation_steps):
    """Evaluate model on the test dataset."""
    # Collect all predictions and true labels
    y_true = []
    y_pred = []
    
    for views, labels in test_dataset.take(validation_steps):
        # Get predictions
        batch_preds = model.predict(views, verbose=0)
        
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

def format_classification_report(report, class_names):
    """
    Given a classification report string and list of class names,
    return a wandb.Table that can be logged. This version also includes 
    summary metrics (accuracy, macro avg, weighted avg).
    """
    report_lines = report.splitlines()
    table_data = []
    columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    
    # Extract per-class rows (assumed to be after header in first two lines)
    # (e.g. lines 3 to 3+len(class_names)-1)
    start = 2
    for line in report_lines[start:start + len(class_names)]:
        parts = line.split()
        if len(parts) < 5:
            # In case the class name contains spaces
            class_name = " ".join(parts[:-4])
            parts = [class_name] + parts[-4:]
        table_data.append(parts)
    
    # Now, go through the remaining lines to get summary rows
    for line in report_lines[start + len(class_names):]:
        if not line.strip():
            continue
        # The accuracy line often only contains 3 pieces: "accuracy", value and support.
        if line.strip().startswith("accuracy"):
            parts = line.split()
            # Format accuracy row as: ["accuracy", "", "", accuracy_value, support]
            if len(parts) == 3:
                table_data.append(["accuracy", "", "", parts[1], parts[2]])
            else:
                table_data.append(parts)
        elif line.strip().startswith("macro avg") or line.strip().startswith("weighted avg"):
            parts = line.split()
            table_data.append(parts)
    
    return wandb.Table(data=table_data, columns=columns)
def visualize_wrong_predictions(model, test_dataset, class_names, num_samples=5, visualize_original=False):
    """Visualize model predictions on only wrongly predicted test samples.
    
    For each wrong sample, only the leftmost (first) view is annotated with the true vs. predicted label.
    
    Args:
         model: Trained model.
         test_dataset: A tf.data.Dataset yielding (views, labels).
         class_names: List of class names.
         num_samples: Maximum number of wrong samples to display.
         visualize_original: If True, display the images as produced by the generator.
    """
    # Take one batch from the test dataset
    for views, labels in test_dataset.take(1):
        preds = model.predict(views, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        
        # Get indexes for wrong predictions.
        wrong_idxs = [i for i, (t, p) in enumerate(zip(true_classes, pred_classes)) if t != p]
        if not wrong_idxs:
            print("No wrong predictions in this batch.")
            return None
        
        wrong_idxs = wrong_idxs[:num_samples]
        num_views = len(views)
        fig = plt.figure(figsize=(20, 4 * len(wrong_idxs)))
        
        for row, i in enumerate(wrong_idxs):
            true_class = true_classes[i]
            pred_class = pred_classes[i]
            for v in range(num_views):
                image = views[v][i].numpy()
                if not visualize_original:
                    # Reverse any preprocessing applied during training, e.g., for ResNet50.
                    image = image + np.array([103.939, 116.779, 123.68])
                    image = image[..., ::-1]  # Convert BGR to RGB if needed.
                image = np.clip(image, 0, 255).astype('uint8')
                
                ax = fig.add_subplot(len(wrong_idxs), num_views, row * num_views + v + 1)
                ax.imshow(image)
                if row == 0:
                    ax.set_title(f"View {v+1}", fontsize=12)
                # On the leftmost view, add an annotation with the true and predicted labels.
                if v == 0:
                    ax.text(5, 25, f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}",
                            fontsize=12, color='yellow', backgroundcolor='black', 
                            verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))
                ax.axis('off')
        plt.tight_layout()
        return fig