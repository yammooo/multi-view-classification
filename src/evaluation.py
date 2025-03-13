import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import SimpleMultiViewDataGenerator
import seaborn as sns

def evaluate_and_log_model(model, output_dir, test_dataset_label, test_ds=None, test_data_dir=None, config=None, class_names=None, validation_steps=None):
    """
    Evaluates the given model on the test dataset and logs results to WandB under unique keys
    prefixed with test_dataset_label to allow multiple evaluations in one run.

    If test_ds (test dataset) is not provided, it is created using test_data_dir and config.
    In that case, validation_steps will be computed automatically from the test generator's sample count.

    Args:
        model (tf.keras.Model): The trained model to evaluate.
        output_dir (str): Directory to save evaluation outputs.
        test_dataset_label (str): A label identifying this test dataset evaluation (e.g., "dataset_A").
        test_ds (tf.data.Dataset, optional): The test dataset. If None, test_data_dir is used to create it.
        test_data_dir (str, optional): Path to test data directory used to create test_ds if not provided.
        config (dict, optional): Configuration dictionary containing keys:
                                 'views', 'input_shape', 'batch_size', etc.
        class_names (list, optional): List of class names. If None and test_ds is not provided,
                                      they are inferred from the generator.
        validation_steps (int, optional): Number of validation steps. If None and test_ds is created here,
                                          it is computed automatically.

    Returns:
        None
    """
    # If test_ds is not supplied, create the generator.
    if test_ds is None:
        if test_data_dir is None:
            raise ValueError("Either test_ds or test_data_dir must be provided.")
        views = config.get("views", ["back_left", "back_right", "front_left", "front_right", "top"]) if config else ["back_left", "back_right", "front_left", "front_right", "top"]
        input_shape = config.get("input_shape", (224, 224, 3)) if config else (224, 224, 3)
        batch_size = config.get("batch_size", 8) if config else 8

        test_gen = SimpleMultiViewDataGenerator(
            data_dir=test_data_dir,
            views=views,
            input_shape=input_shape,
            batch_size=batch_size,
            test_split=1.0
        )
        test_ds = test_gen.get_test_dataset()
        if class_names is None:
            class_names = test_gen.get_class_names()
        # Automatically compute validation_steps from the generator sample count if not provided.
        if validation_steps is None:
            validation_steps = len(test_gen.test_samples) // batch_size
            if len(test_gen.test_samples) % batch_size != 0:
                validation_steps += 1

    # Evaluate the model.
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)
    pred_wrong_fig = plot_wrong_predictions(model, test_ds, class_names)
    
    # Log unique keys using the test_dataset_label.
    wandb.log({
        f"{test_dataset_label}/classification_report": format_classification_report(report, class_names),
        f"{test_dataset_label}/confusion_matrix": wandb.Image(cm_fig),
        f"{test_dataset_label}/wrong_predictions": wandb.Image(pred_wrong_fig),
    })

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
    
    # Extract per-class rows
    start = 2
    for line in report_lines[start:start + len(class_names)]:
        parts = line.split()
        if len(parts) < 5:
            # In case the class name contains spaces
            class_name = " ".join(parts[:-4])
            parts = [class_name] + parts[-4:]
        table_data.append(parts)
    
    # Process summary rows.
    for line in report_lines[start + len(class_names):]:
        if not line.strip():
            continue
        if line.strip().startswith("accuracy"):
            parts = line.split()
            # Format accuracy row as: ["accuracy", "", "", accuracy_value, support]
            if len(parts) == 3:
                table_data.append(["accuracy", "", "", parts[1], parts[2]])
            else:
                table_data.append(parts)
        elif line.strip().startswith("macro avg") or line.strip().startswith("weighted avg"):
            parts = line.split()
            # If splitting produces 6 tokens, merge the first two.
            if len(parts) == 6:
                parts = [" ".join(parts[:2])] + parts[2:]
            table_data.append(parts)
    
    return wandb.Table(data=table_data, columns=columns)

def plot_wrong_predictions(model, test_dataset, class_names, num_samples=5, visualize_original=True):
    """
    Generate a figure that visualizes wrong predictions from the test dataset.
    
    For each wrong sample, only the leftmost (first) view is annotated with the true vs. predicted label.
    If no wrong predictions exist, the figure will display a message indicating so.
    
    Args:
         model: Trained model.
         test_dataset: A tf.data.Dataset yielding (views, labels).
         class_names: List of class names.
         num_samples: Maximum number of wrong samples to display.
         visualize_original: If True, displays images as produced by the generator.
         
    Returns:
         matplotlib.figure.Figure: The figure containing the visualization.
    """
    # Take one batch from the test dataset.
    for views, labels in test_dataset.take(1):
        preds = model.predict(views, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        
        # Get indexes for wrong predictions.
        wrong_idxs = [i for i, (t, p) in enumerate(zip(true_classes, pred_classes)) if t != p]
        
        # If there are no wrong predictions, create a figure with a message.
        if not wrong_idxs:
            fig = plt.figure(figsize=(10, 4))
            ax = fig.add_subplot(1, 1, 1)
            ax.text(0.5, 0.5, "No wrong predictions in this batch.", 
                    ha="center", va="center", fontsize=14)
            ax.axis("off")
            return fig
        
        # Limit number of wrong predictions to num_samples.
        wrong_idxs = wrong_idxs[:num_samples]
        num_views = len(views)
        fig = plt.figure(figsize=(20, 4 * len(wrong_idxs)))
        
        for row, i in enumerate(wrong_idxs):
            true_class = true_classes[i]
            pred_class = pred_classes[i]
            for v in range(num_views):
                image = views[v][i].numpy()
                if not visualize_original:
                    # Reverse any preprocessing, e.g., for ResNet50.
                    image = image + np.array([103.939, 116.779, 123.68])
                    image = image[..., ::-1]  # Convert BGR to RGB if needed.
                image = np.clip(image, 0, 255).astype('uint8')
                
                ax = fig.add_subplot(len(wrong_idxs), num_views, row * num_views + v + 1)
                ax.imshow(image)
                if row == 0:
                    ax.set_title(f"View {v+1}", fontsize=12)
                # On the leftmost view, add annotation with true and predicted labels.
                if v == 0:
                    ax.text(5, 25, f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}",
                            fontsize=12, color='yellow', backgroundcolor='black',
                            verticalalignment='top', bbox=dict(facecolor='black', alpha=0.5))
                ax.axis('off')
        plt.tight_layout()
        return fig