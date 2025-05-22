import numpy as np
import random
import torch

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_stream_generator(X_id, y_id, X_ood, y_ood, batch_size=100, device="cpu", seed=42):
    """
    Data stream generator that combines ID and OOD data, shuffles, and yields batches in a deterministic manner.

    Parameters:
        X_id (np.ndarray): Features of ID samples.
        y_id (np.ndarray): Labels of ID samples.
        X_ood (np.ndarray): Features of OOD samples.
        y_ood (np.ndarray): Labels of OOD samples.
        batch_size (int): Size of each batch.
        device (str): Device for tensors ("cpu" or "cuda").
        seed (int): Random seed for reproducibility.

    Yields:
        torch.Tensor, torch.Tensor: A batch of features and labels.
    """
    # Combine ID and OOD data
    X_combined = np.vstack([X_id, X_ood])
    y_combined = np.concatenate([y_id, y_ood])


    # Create a unified index for shuffling
    combined_indices = np.arange(len(X_combined))

    # Shuffle indices with a fixed seed
    rng = np.random.default_rng(seed)  # Use NumPy's random generator with a fixed seed
    rng.shuffle(combined_indices)

    print(f"Total samples: {len(combined_indices)}")

    # Split data into shuffled batches
    for i in range(0, len(combined_indices), batch_size):
        print(f"Batch start index: {i}, Batch size: {batch_size}")
        # Select batch indices
        batch_indices = combined_indices[i:i + batch_size]
        
        # Extract batch data
        X_batch = X_combined[batch_indices]
        y_batch = y_combined[batch_indices]

        # Convert to tensors and yield
        yield torch.tensor(X_batch, dtype=torch.float32, device=device), y_batch
