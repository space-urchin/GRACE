import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os


def calculate_accuracy(y_true, y_pred):
    """Calculate accuracy from predicted and true labels."""
    _, predicted_labels = torch.max(y_pred, 1)
    _, true_labels = torch.max(y_true, 1)
    return (predicted_labels == true_labels).float().mean().item()


def evaluate_on_combined_test_set(
    model, encoder, X_test, y_test,old_classes, new_classes, results_file, seed_dir, device="cpu"
):

    # Reset indices for consistency
    y_test = y_test.reset_index(drop=True)
    X_test = pd.DataFrame(X_test).reset_index(drop=True)

    # Get the list of known categories from the encoder
    known_categories = set(encoder.categories_[0])

    # Filter out instances with unknown labels
    filtered_indices = [i for i, label in enumerate(y_test) if label in known_categories]
    y_test_filtered = y_test.iloc[filtered_indices]
    X_test_filtered = X_test.iloc[filtered_indices].values

    # Convert class labels to strings for the classification report
    target_names = [str(cat) for cat in encoder.categories_[0]]

    # One-hot encode the filtered labels
    y_test_filtered_array = y_test_filtered.to_numpy()
    y_test_onehot_filtered = torch.tensor(
        encoder.transform(y_test_filtered_array.reshape(-1, 1)),
        dtype=torch.float32,
        device=device
    )

    # Convert filtered data to tensors
    X_test_tensor = torch.tensor(X_test_filtered, dtype=torch.float32, device=device)

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_test_pred, _, _ = model(X_test_tensor)
        combined_accuracy = calculate_accuracy(y_test_onehot_filtered, y_test_pred)
        print(f"Filtered Test Accuracy (Ignoring Unknown Labels): {combined_accuracy * 100:.2f}%")

        # Extract predicted and true labels
        y_test_pred_labels = torch.argmax(y_test_pred, dim=1).cpu().numpy()
        y_test_true_labels = torch.argmax(y_test_onehot_filtered, dim=1).cpu().numpy()
        y_test_pred_probs = y_test_pred.cpu().numpy()

        # Generate classification report
        class_report = classification_report(
            y_test_true_labels, y_test_pred_labels, 
            target_names=target_names, 
            output_dict=True
        )

        # Print full classification report
        print("Classification Report (Filtered Test Set):")
        print(classification_report(
            y_test_true_labels, y_test_pred_labels, 
            target_names=target_names
        ))

        # Generate confusion matrix
        conf_matrix = confusion_matrix(y_test_true_labels, y_test_pred_labels)
        print("\nConfusion Matrix:")
        print(conf_matrix)

        # Save confusion matrix as a CSV
        conf_matrix_df = pd.DataFrame(
            conf_matrix,
            index=target_names,
            columns=target_names
        )
        conf_matrix_csv_path = os.path.join(seed_dir, f"confusion_matrix.csv")
        conf_matrix_df.to_csv(conf_matrix_csv_path)
        print(f"Confusion matrix saved to {conf_matrix_csv_path}")

        # Save confusion matrix as a heatmap image
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix_df, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        conf_matrix_image_path = os.path.join(seed_dir, f"confusion_matrix.png")
        plt.savefig(conf_matrix_image_path)
        plt.close()
        print(f"Confusion matrix heatmap saved to {conf_matrix_image_path}")

        # Calculate F1-scores for old and new classes
        print("\nF1-scores for each class:")
        old_f1_scores = []
        new_f1_scores = []


        for class_name, metrics in class_report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                try:
                    f1_score = metrics['f1-score']
                    print(f"Class {class_name}: F1-score = {f1_score:.2f}")
                    if class_name in old_classes:
                        old_f1_scores.append(f1_score)
                    elif class_name in new_classes:
                        new_f1_scores.append(f1_score)
                except ValueError:
                    continue 

        # Calculate macro averages
        if old_f1_scores:  # Only calculate if there are scores
            old_macro_avg = np.sum(old_f1_scores) / len(old_classes)
            print(f"\nMacro Average F1-score for Old Classes: {old_macro_avg:.2f}")
        else:
            old_macro_avg = 0
            print("\nNo F1-scores available for Old Classes")

        if new_f1_scores:  # Only calculate if there are scores
            new_macro_avg = np.sum(new_f1_scores) / len(new_classes)
            print(f"Macro Average F1-score for New Classes: {new_macro_avg:.2f}")
        else:
            new_macro_avg = 0
            print("No F1-scores available for New Classes")

        # Calculate overall F1 score (macro average)
        overall_f1 = np.mean([old_macro_avg, new_macro_avg])
        print(f"Overall Macro F1-score: {overall_f1:.2f}")

        # Save the results to the results file
        with open(results_file, "a") as f:
            f.write(f"{old_macro_avg:.4f},{new_macro_avg:.4f},{overall_f1:.4f},{combined_accuracy:.4f}\n")

        return old_macro_avg, new_macro_avg, overall_f1, combined_accuracy
    


def plot_results(results, results_dir):
    """Create visualizations of results across seeds."""
    seeds = [r['seed'] for r in results]
    
    # Create performance comparison chart
    plt.figure(figsize=(12, 8))
    
    # Plot metrics
    plt.plot(seeds, [r['old_f1'] for r in results], 'o-', label='Old Classes F1', linewidth=2)
    plt.plot(seeds, [r['new_f1'] for r in results], 'o-', label='New Classes F1', linewidth=2)
    plt.plot(seeds, [r['overall_f1'] for r in results], 'o-', label='Overall F1', linewidth=2)
    plt.plot(seeds, [r['accuracy'] for r in results], 'o-', label='Accuracy', linewidth=2)
    
    plt.xlabel('Seed Value')
    plt.ylabel('Performance')
    plt.title('Performance Metrics Across Different Seeds')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(seeds)
    
    # Save the plot
    plt.savefig(f"{results_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create separate bar charts for each metric
    metrics = ['old_f1', 'new_f1', 'overall_f1', 'accuracy']
    titles = ['Old Classes F1', 'New Classes F1', 'Overall F1', 'Accuracy']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        values = [r[metric] for r in results]
        axes[i].bar(seeds, values, color='skyblue', edgecolor='navy')
        axes[i].set_title(title)
        axes[i].set_xlabel('Seed')
        axes[i].set_ylabel('Value')
        axes[i].grid(True, linestyle='--', alpha=0.7)
        axes[i].set_ylim(0, 1.0)  # Set y-axis limits for consistency
        
        # Add value labels on top of bars
        for j, v in enumerate(values):
            axes[i].text(seeds[j], v + 0.02, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/metrics_by_seed.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a correlation heatmap between metrics
    correlation_data = {
        'Old Classes F1': [r['old_f1'] for r in results],
        'New Classes F1': [r['new_f1'] for r in results],
        'Overall F1': [r['overall_f1'] for r in results],
        'Accuracy': [r['accuracy'] for r in results],
        'Training Time': [r['training_time'] for r in results]
    }
    
    corr_df = pd.DataFrame(correlation_data)
    corr_matrix = corr_df.corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Between Metrics')
    plt.tight_layout()
    plt.savefig(f"{results_dir}/metrics_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()
