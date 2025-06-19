import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import math
import os

import time
from datetime import datetime
from utils import set_seed, data_stream_generator
from preparedata import prepare_data
from model import MLP, CrossEntropyLoss, retrain_model
from grace import hybrid_instance_selection
from evaluate import evaluate_on_combined_test_set, plot_results

# Define seeds to run
SEEDS = [50, 60, 70]
k_uncertain = 50

# Create results directory
results_dir = "../outputs/aci/gmm_k" + str(k_uncertain) + "_time_" + datetime.now().strftime("%Y%m%d_%H%M%S")
os.makedirs(results_dir, exist_ok=True)

# Create summary file
summary_file = f"{results_dir}/summary_results.csv"
with open(summary_file, "w") as f:
    f.write("seed,old_classes_f1,new_classes_f1,overall_f1,accuracy,training_time\n")

# Define dataset, old and new classes
# dataset_name = "iot"
# old_classes = ['Port Scan', 'Benign', 'ICMP Flood', 'Ping Sweep', 'DNS Flood','Vulnerability Scan', 'OS Scan'  ]
# new_classes = ["Slowloris", "Dictionary Attack", "UDP Flood", "SYN Flood"]
# streamsize = 20000

dataset_name = "unsw"
old_classes = ['Generic', 'Normal', 'Exploits']
new_classes = ['DoS', 'Reconnaissance', 'Fuzzers']
streamsize = 1000

# dataset_name = "cic"
# old_classes = ['Normal Traffic', 'DoS', 'DDoS', 'Port Scanning']
# new_classes = ['Web Attacks', 'Bots', 'Brute Force']
# streamsize = 10000





def active_learning(
    model, encoder, X_labeled, y_labeled, X_unlabeled_id, y_unlabeled_id,
    X_train_ood, y_train_ood, X_val_tensor, y_val_tensor, datastream, stream_size,
    num_instances, X_test, y_test, best_model_path, results_file, seed_dir, device="cpu"
):
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.to(device)
    
    # Ensure categories are sorted initially
    existing_classes = sorted(encoder.categories_[0].tolist())
    
    # Create data stream
    data_stream = data_stream_generator(
        X_unlabeled_id, y_unlabeled_id, 
        X_train_ood, y_train_ood, 
        batch_size=stream_size,
        device=device
    )
    
    for stream_idx in range(datastream):
        print(f"\nProcessing data stream {stream_idx + 1}/{datastream}")
        batchX, batchY = next(data_stream)

        # Convert labels to numeric format for GMM
        label_mapping = {label: idx for idx, label in enumerate(existing_classes)}
        numeric_y_labeled = np.array([label_mapping[label] for label in y_labeled])
        # Select instances based on entropy
        selected_data = hybrid_instance_selection(
            batchX, batchY, model, X_labeled, numeric_y_labeled,
            k_uncertain, 
            num_instances=num_instances, 
            device=device,
            method='gmm'
        )
        # Select instances based on entropy
        X_selected = [x[0] for x in selected_data]
        y_selected = [x[1] for x in selected_data]

        # Check for new classes
        new_classes = sorted(list(set([label for label in y_selected if label not in existing_classes])))
        
        if new_classes:
            print(f"New classes detected: {new_classes}")
            # Update existing classes while maintaining sorted order
            existing_classes = sorted(list(set(existing_classes + new_classes)))
            
            # Create new encoder with sorted categories
            encoder = OneHotEncoder(sparse_output=False)
            # Reshape and fit with sorted categories
            encoder.fit(np.array(existing_classes).reshape(-1, 1))

            # Update model's output layer
            output_dim = len(existing_classes)
            old_weights = model.output_layer.weight.data
            old_bias = model.output_layer.bias.data
            
            # Create new output layer
            new_output_layer = nn.Linear(model.output_layer.in_features, output_dim).to(device)
            
            # Copy existing weights for old classes
            with torch.no_grad():
                new_output_layer.weight.data[:len(old_weights)] = old_weights
                new_output_layer.bias.data[:len(old_bias)] = old_bias
            
            # Update model's output layer
            model.output_layer = new_output_layer
            print(f"Model updated to handle {output_dim} classes.")

        # Update labeled data
        y_labeled = np.concatenate([y_labeled, y_selected])
        y_labeled_encoded = torch.tensor(
            encoder.transform(np.array(y_labeled).reshape(-1, 1)), 
            dtype=torch.float32, 
            device=device
        )
        X_labeled = torch.cat([X_labeled, torch.stack(X_selected)], dim=0)

        # Compute class weights
        unique_classes = np.array(sorted(np.unique(y_labeled)))
        class_weights = compute_class_weight(
            class_weight="balanced", 
            classes=unique_classes,
            y=y_labeled
        )
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

        # Create DataLoader for training
        labeled_dataset = TensorDataset(X_labeled, y_labeled_encoded)
        labeled_loader = DataLoader(labeled_dataset, batch_size=256, shuffle=True)

        # Retrain model
        model = retrain_model(
            model=model, 
            optimizer=optimizer, 
            data_loader=labeled_loader, 
            class_weights=class_weights_tensor, 
            epochs=100,
            X_val_tensor=X_val_tensor,
            y_val_tensor=y_val_tensor,
            best_model_path=best_model_path,
            device=device
        )

        print(f"Data Stream {stream_idx + 1}/{datastream} processed and model retrained.")

        # Periodic evaluation
        if (stream_idx + 1) % 10 == 0:
            print(f"\nEvaluating on Test Set after {stream_idx + 1} data stream iterations...")
            evaluate_on_combined_test_set(
                model=model,
                encoder=encoder,
                X_test=X_test,
                y_test=y_test,
                old_classes = old_classes,
                new_classes = new_classes,
                results_file=results_file,
                seed_dir = seed_dir,
                device=device,
            )

        # Print class distribution
        unique_values, counts = np.unique(y_labeled, return_counts=True)
        print("\nCurrent class distribution:")
        for val, count in zip(unique_values, counts):
            print(f"Class {val}: {count} samples")
        print(f"Total samples: {len(y_labeled)}")

    return model, encoder



# ------------------ Main Function ------------------ #

def run_with_seed(seed, dataset_name):
    print(f"\n{'='*50}")
    print(f"RUNNING WITH SEED {seed} on {dataset_name.upper()}")
    print(f"{'='*50}\n")

    seed_dir = os.path.join(results_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    best_path = os.path.join(seed_dir, f"best_model_{seed}.pth")
    results_file = os.path.join(seed_dir, f"results_{seed}.csv")

    with open(results_file, "w") as f:
        f.write("old_f1,new_f1,overall_f1,accuracy\n")

    start_time = time.time()
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_id, y_id, X_ood, y_ood, X_test, y_test, label_col = prepare_data(dataset_name, old_classes, new_classes)

    X_labeled, X_unlabeled_id, y_labeled, y_unlabeled_id = train_test_split(
        X_id, y_id, test_size=0.99, stratify=y_id, random_state=42
    )

    X_unlabeled_id, X_val_id, y_unlabeled_id, y_val_id = train_test_split(
        X_unlabeled_id, y_unlabeled_id, test_size=0.1, stratify=y_unlabeled_id, random_state=42
    )

    encoder = OneHotEncoder(sparse_output=False)
    y_train_encoded = encoder.fit_transform(y_labeled.values.reshape(-1, 1))
    y_val_encoded = encoder.transform(y_val_id.values.reshape(-1, 1))

    class_weights = compute_class_weight("balanced", classes=np.unique(y_labeled), y=y_labeled)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    X_train_tensor = torch.tensor(X_labeled, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val_id, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.float32, device=device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # Model setup
    model = MLP(input_dim=X_train_tensor.shape[1], output_dim=y_train_tensor.shape[1])
    loss_function = CrossEntropyLoss(class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    model.to(device)

    # Early stopping parameters
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting Training with Early Stopping...")

    print(X_labeled.shape)

    for epoch in range(100):
        # Training loop
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred, embeddings, logits = model(X_batch)
            loss = loss_function(y_batch, y_pred)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation loop
        model.eval()
        with torch.no_grad():
            y_val_pred, _, _ = model(X_val_tensor)
            val_loss = loss_function(y_val_tensor, y_val_pred).item()

        print(f"Epoch {epoch + 1}: Train Loss: {np.mean(train_losses):.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load best model
    model.load_state_dict(torch.load(best_path))
    print("Training completed with Early Stopping.")

    datastream = math.ceil((X_unlabeled_id.shape[0] + X_ood.shape[0]) / streamsize)
    print(f"Number of datastreams: {datastream}")

    # Start active learning
    model, encoder = active_learning(
        model=model,
        encoder=encoder,
        X_labeled=X_train_tensor,
        y_labeled=y_labeled,
        X_unlabeled_id=X_unlabeled_id,
        y_unlabeled_id=y_unlabeled_id,
        X_train_ood=X_ood,
        y_train_ood=y_ood,
        X_val_tensor=X_val_tensor,
        y_val_tensor=y_val_tensor,
        datastream=datastream,
        stream_size=streamsize,
        num_instances=100,
        X_test=X_test,
        y_test=y_test[label_col],
        best_model_path=best_path,
        results_file=results_file,
        seed_dir=seed_dir,
        device=device,
    )


    # Print data shapes
    print("\nData shapes:")
    print(f"Labeled data: {X_labeled.shape}")
    print(f"Unlabeled ID data: {X_unlabeled_id.shape}")
    print(f"OOD data: {X_ood.shape}")
    print(f"Test data: {X_test.shape}")
    print(f"Test labels: {y_test.shape}")

    # Final evaluation
    print("\nFinal Evaluation on Test Set...")
    old_f1, new_f1, overall_f1, accuracy = evaluate_on_combined_test_set(
        model=model,
        encoder=encoder,
        X_test=X_test,
        y_test=y_test[label_col],
        old_classes=old_classes,
        new_classes=new_classes,
        results_file=results_file,
        seed_dir=seed_dir,
        device=device
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Update summary file
    with open(summary_file, "a") as f:
        f.write(f"{seed},{old_f1:.4f},{new_f1:.4f},{overall_f1:.4f},{accuracy:.4f},{training_time:.2f}\n")
    
    return old_f1, new_f1, overall_f1, accuracy, training_time


def main():
    """Main function to run experiments with multiple seeds."""
    print(f"Starting experiments with seeds: {SEEDS}")
    print(f"Results will be saved to: {results_dir}")

    print(dataset_name)
    
    all_results = []
    
    # Run with each seed
    for seed in SEEDS:
        try:
            old_f1, new_f1, overall_f1, accuracy, training_time = run_with_seed(seed, dataset_name=dataset_name)
            all_results.append({
                'seed': seed,
                'old_f1': old_f1,
                'new_f1': new_f1,
                'overall_f1': overall_f1,
                'accuracy': accuracy,
                'training_time': training_time
            })
        except Exception as e:
            print(f"Error running with seed {seed}: {e}")
    
    # Calculate and display statistics
    if all_results:
        print("\n\n" + "="*50)
        print("SUMMARY OF RESULTS ACROSS ALL SEEDS")
        print("="*50)
        
        # Calculate mean and standard deviation
        old_f1_values = [r['old_f1'] for r in all_results]
        new_f1_values = [r['new_f1'] for r in all_results]
        overall_f1_values = [r['overall_f1'] for r in all_results]
        accuracy_values = [r['accuracy'] for r in all_results]
        
        print(f"Old Classes F1: {np.mean(old_f1_values):.4f} ± {np.std(old_f1_values):.4f}")
        print(f"New Classes F1: {np.mean(new_f1_values):.4f} ± {np.std(new_f1_values):.4f}")
        print(f"Overall F1: {np.mean(overall_f1_values):.4f} ± {np.std(overall_f1_values):.4f}")
        print(f"Accuracy: {np.mean(accuracy_values):.4f} ± {np.std(accuracy_values):.4f}")
        
        # Save summary statistics
        with open(f"{results_dir}/final_statistics.txt", "w") as f:
            f.write("SUMMARY OF RESULTS ACROSS ALL SEEDS\n")
            f.write("="*50 + "\n")
            f.write(f"Seeds: {SEEDS}\n\n")
            f.write(f"Old Classes F1: {np.mean(old_f1_values):.4f} ± {np.std(old_f1_values):.4f}\n")
            f.write(f"New Classes F1: {np.mean(new_f1_values):.4f} ± {np.std(new_f1_values):.4f}\n")
            f.write(f"Overall F1: {np.mean(overall_f1_values):.4f} ± {np.std(overall_f1_values):.4f}\n")
            f.write(f"Accuracy: {np.mean(accuracy_values):.4f} ± {np.std(accuracy_values):.4f}\n")
            
        # Create visualization of results
        plot_results(all_results, results_dir)
    
    print(f"\nAll experiments completed. Results saved to {results_dir}")




if __name__ == "__main__":
    main()