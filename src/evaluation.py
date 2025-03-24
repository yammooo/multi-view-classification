import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
from sklearn.metrics import classification_report, confusion_matrix
from data_generator import SimpleMultiViewDataGenerator
import seaborn as sns

def evaluate_and_log_model(model, output_dir, test_dataset_label, test_ds=None, test_data_dir=None, config=None, class_names=None, validation_steps=None):
    """
    Evaluates the given model on the test dataset and logs results to WandB.
    
    Overall metrics (classification report, confusion matrix) are logged.
    Additionally, for one batch from the test dataset, per-sample predictions (5-view image and prediction histogram)
    are aggregated into a wandb.Table and logged.
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
        if validation_steps is None:
            validation_steps = len(test_gen.test_samples) // batch_size
            if len(test_gen.test_samples) % batch_size != 0:
                validation_steps += 1

    # Evaluate overall metrics.
    report, cm_fig, y_true, y_pred = evaluate_model(model, test_ds, class_names, validation_steps)
    wandb.log({
        f"{test_dataset_label}/classification_report": format_classification_report(report, class_names),
        f"{test_dataset_label}/confusion_matrix": wandb.Image(cm_fig)
    })

    # Create a table for sample-level predictions.
    sample_table = wandb.Table(columns=["Sample", "True Label", "Predicted Label", "Views", "Histogram"])
    
    # Process one batch from the test dataset.
    for views, labels in test_ds.take(1):
        preds = model.predict(views, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        true_classes = np.argmax(labels.numpy(), axis=1)
        num_views = len(views)
        num_samples = len(true_classes)

        for i in range(num_samples):
            # Build the multi-view image figure.
            sample_views = [views[v][i].numpy() for v in range(num_views)]
            fig_views = plot_sample_views(sample_views, true_classes[i], pred_classes[i], class_names)
            # Build the histogram figure of softmax scores.
            fig_hist = plot_sample_histogram(preds[i], class_names)
            # Add row to the table.
            sample_table.add_data(i, class_names[true_classes[i]], class_names[pred_classes[i]],
                                    wandb.Image(fig_views),
                                    wandb.Image(fig_hist))
    
    wandb.log({f"{test_dataset_label}/sample_table": sample_table})

def plot_confusion_matrix_from_cm(cm, class_names):
    """
    Plot a confusion matrix using seaborn with modifications for readability.
    """
    fig, ax = plt.subplots(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names,
                ax=ax,
                annot_kws={"fontsize": 6})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    return fig

def evaluate_model(model, test_dataset, class_names, validation_steps):
    """Evaluate model on the test dataset."""
    y_true = []
    y_pred = []
    for views, labels in test_dataset.take(validation_steps):
        batch_preds = model.predict(views, verbose=0)
        batch_pred_classes = np.argmax(batch_preds, axis=1)
        batch_true_classes = np.argmax(labels.numpy(), axis=1)
        y_true.extend(batch_true_classes)
        y_pred.extend(batch_pred_classes)
    all_labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, target_names=class_names, labels=all_labels, zero_division=0)
    print("Classification Report:")
    print(report)
    cm = confusion_matrix(y_true, y_pred, labels=all_labels)
    cm_fig = plot_confusion_matrix_from_cm(cm, class_names)
    return report, cm_fig, y_true, y_pred

def format_classification_report(report, class_names):
    """
    Given a classification report string and a list of class names,
    return a wandb.Table that can be logged.
    """
    report_lines = report.splitlines()
    table_data = []
    columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
    start = 2
    for line in report_lines[start:start + len(class_names)]:
        parts = line.split()
        if len(parts) < 5:
            class_name = " ".join(parts[:-4])
            parts = [class_name] + parts[-4:]
        table_data.append(parts)
    for line in report_lines[start + len(class_names):]:
        if not line.strip():
            continue
        if line.strip().startswith("accuracy"):
            parts = line.split()
            if len(parts) == 3:
                table_data.append(["accuracy", "", "", parts[1], parts[2]])
            else:
                table_data.append(parts)
        elif line.strip().startswith("macro avg") or line.strip().startswith("weighted avg"):
            parts = line.split()
            if len(parts) == 6:
                parts = [" ".join(parts[:2])] + parts[2:]
            table_data.append(parts)
    return wandb.Table(data=table_data, columns=columns)

def plot_sample_views(sample_views, true_class, pred_class, class_names):
    """
    Create a figure for one sample showing its 5-view images.
    The first view is annotated with the true and predicted labels.
    """
    num_views = len(sample_views)
    fig = plt.figure(figsize=(5 * num_views, 4))
    for j, image in enumerate(sample_views):
        ax = fig.add_subplot(1, num_views, j + 1)
        image = np.clip(image, 0, 255).astype('uint8')
        ax.imshow(image)
        ax.axis("off")
        if j == 0:
            color = "green" if true_class == pred_class else "red"
            ax.set_title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}", color=color, fontsize=12)
    plt.tight_layout()
    return fig

def plot_sample_histogram(sample_probs, class_names):
    """
    Create a Matplotlib bar plot displaying the prediction probabilities
    for each class for the given sample.
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(range(len(class_names)), sample_probs, color="grey")
    pred_class = int(np.argmax(sample_probs))
    bars[pred_class].set_color("blue")
    for j, prob in enumerate(sample_probs):
        ax.text(j, prob, f"{prob:.2f}", ha="center", va="bottom", fontsize=6, rotation=90)
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_title("Prediction Score Distribution", fontsize=10)
    plt.tight_layout()
    return fig