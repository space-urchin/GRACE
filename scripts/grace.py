import torch
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances_argmin_min

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]

def select_bandwidth_median(embeddings: torch.Tensor, sample_size: int = 1000) -> float:
    n = embeddings.size(0)
    if n <= 1:
        return 1.0
    idx = torch.randperm(n)[: min(n, sample_size)]
    sample = embeddings[idx]
    dists = torch.cdist(sample, sample, p=2.0)
    iu = torch.triu_indices(dists.size(0), dists.size(1), offset=1)
    m = dists[iu[0], iu[1]].median().item()
    return m if m > 0 else 1.0

def compute_kde_density_epanechnikov(batch_emb, support_emb, bandwidth):
    dist2 = torch.cdist(batch_emb, support_emb, p=2.0).pow(2)
    u2 = dist2 / (bandwidth ** 2)
    K = torch.clamp(1.0 - u2, min=0.0)
    return K.mean(dim=1)

def centered_cov_torch(x):
    n = x.shape[0]
    if n <= 1:
        return torch.eye(x.shape[1], device=x.device)
    return 1 / (n - 1) * x.t().mm(x)

def gmm_fit(embeddings, labels, num_classes):
    with torch.no_grad():
        classwise_mean = torch.stack([torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)])
        classwise_cov = torch.stack([centered_cov_torch(embeddings[labels == c] - classwise_mean[c]) for c in range(num_classes)])
    gmm = None
    with torch.no_grad():
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(classwise_cov.shape[1], device=classwise_cov.device).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(loc=classwise_mean, covariance_matrix=classwise_cov + jitter)
                break
            except (RuntimeError, ValueError) as e:
                if "cholesky" in str(e) or "covariance_matrix has invalid values" in str(e):
                    continue
        if gmm is None:
            print("Warning: Using large jitter value for numerical stability")
            jitter = torch.eye(classwise_cov.shape[1], device=classwise_cov.device).unsqueeze(0)
            gmm = torch.distributions.MultivariateNormal(loc=classwise_mean, covariance_matrix=classwise_cov + jitter)
    return gmm

def compute_gmm_density(gmm, embeddings, num_classes):
    log_probs = gmm.log_prob(embeddings[:, None, :])
    probs = torch.exp(log_probs)
    class_priors = torch.ones(num_classes, device=embeddings.device) / num_classes
    return torch.sum(probs * class_priors, dim=1)

def compute_gmm_density(gmm, embeddings, class_priors):
    """
    Compute GMM density: q(z) = Σy q(z|y) * π(y)
    """
    log_probs = gmm.log_prob(embeddings[:, None, :])  # Shape: [B, C]
    probs = torch.exp(log_probs)                      # q(z|y)
    density = torch.sum(probs * class_priors, dim=1)  # q(z)
    return density

def class_probs_from_labels(labels, num_classes):
    """
    Compute class probabilities from labeled tensor.
    """
    class_counts = torch.tensor([(labels == c).sum() for c in range(num_classes)], dtype=torch.float32, device=labels.device)
    class_probs = class_counts / class_counts.sum()
    print(class_counts)
    return class_probs

def coreset_selection(batch_emb, seed_emb, rem_X, rem_y, num_instances):
    selected = []
    seed_np = seed_emb.cpu().numpy()
    rem_emb_np = batch_emb.cpu().numpy()
    rem_idx = np.arange(len(rem_emb_np))
    centers = seed_np.copy()
    n_add = min(num_instances - len(centers), len(rem_emb_np))

    for _ in range(n_add):
        _, dists = pairwise_distances_argmin_min(rem_emb_np, centers)
        pick = int(np.argmax(dists))
        selected.append((rem_X[pick], rem_y[pick]))
        centers = np.vstack([centers, rem_emb_np[pick : pick + 1]])
        rem_emb_np = np.delete(rem_emb_np, pick, axis=0)
        rem_idx = np.delete(rem_idx, pick)
        rem_X = rem_X[torch.arange(len(rem_X)) != pick]
        rem_y.pop(pick)

    return selected

def hybrid_instance_selection(batchX, batchY, model, X_labeled, y_labeled, k_uncertain, num_instances=10, device="cpu", method="kde"):
    model.eval()
    X_data = torch.stack([x for x in batchX]).to(device)
    y_data = [y for y in batchY]

    with torch.no_grad():
        _, batch_emb, _ = model(X_data)
        _, labeled_emb, _ = model(X_labeled.to(device))

    if method == "kde":
        h = select_bandwidth_median(labeled_emb)
        density = compute_kde_density_epanechnikov(batch_emb, labeled_emb, h)
    elif method == "gmm":
        num_classes = len(np.unique(y_labeled))
        labels_tensor = torch.tensor(y_labeled, device=device)
        class_priors = class_probs_from_labels(labels_tensor, num_classes)
        gmm = gmm_fit(labeled_emb, torch.tensor(y_labeled, device=device), num_classes)
        density = compute_gmm_density(gmm, batch_emb, class_priors)
    else:
        raise ValueError("Invalid method: choose 'kde' or 'gmm'")

    uncertain_idx = torch.topk(density, k=k_uncertain, largest=False).indices.cpu().tolist()
    selected = [(X_data[i], y_data[i]) for i in uncertain_idx]

    mask = torch.ones(len(X_data), dtype=torch.bool)
    mask[uncertain_idx] = False
    rem_X = X_data[mask]
    rem_y = [y_data[i] for i in range(len(y_data)) if mask[i]]
    rem_emb = batch_emb[mask]
    unc_emb = batch_emb[uncertain_idx]

    selected.extend(coreset_selection(rem_emb, unc_emb, rem_X, rem_y, num_instances))
    return selected
