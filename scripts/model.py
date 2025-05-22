import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(2048, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(256, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)

        self.embeddings_layer = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, output_dim)

    def forward(self, x):
        if x.size(0) > 1:
            x = F.relu(self.bn1(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = self.dropout1(x)

        if x.size(0) > 1:
            x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        if x.size(0) > 1:
            x = F.relu(self.bn3(self.fc3(x)))
        else:
            x = F.relu(self.fc3(x))
        x = self.dropout3(x)

        embeddings = F.relu(self.embeddings_layer(x))
        logits = self.output_layer(embeddings)
        probabilities = F.softmax(logits, dim=1)
        return probabilities, embeddings, logits


class CrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(CrossEntropyLoss, self).__init__()
        self.cross_entropy_loss = nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, y_true, y_pred, embeddings=None):
        return self.cross_entropy_loss(y_pred, y_true.argmax(dim=1))
    


def retrain_model(model, optimizer, data_loader, class_weights, epochs, X_val_tensor, y_val_tensor, best_model_path, device="cpu"):
    """Retrain the ensemble model with updated labeled data"""
    criterion = CrossEntropyLoss(class_weights)
    
    # Early stopping parameters
    patience = 5
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_losses = []
        model.train()
        
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            y_pred, _, _ = model(X_batch)
            loss = criterion(y_batch, y_pred)
            epoch_losses.append(loss.item())
            
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            y_val_pred, _, _ = model(X_val_tensor)
            val_loss = criterion(y_val_tensor, y_val_pred).item()

        print(f"Epoch {epoch + 1}: Train Loss: {np.mean(epoch_losses):.4f}, Val Loss: {val_loss:.4f}")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    # Load the best model
    model.load_state_dict(torch.load(best_model_path))
    return model